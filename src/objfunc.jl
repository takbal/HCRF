# evaluate the objective function (called by Optim.jl)
function (obj::ObjectiveFunc)(_, G, parameters::Vector{Float64})

    if !isnothing(G)
        fill!(G, 0)
    end

    n_features = obj.model.state_parameters_shape[3]

    state_parameters = reshape( view(parameters, 1:obj.model.state_parameters_count), obj.model.state_parameters_shape )
    transition_parameters = view(parameters, obj.model.state_parameters_count+1:(obj.model.state_parameters_count + length(obj.model.transitions) ) )
    feat_dot_parameters = reshape(state_parameters, :, n_features) * obj.features # (states * classes) x time_steps matrix

    for thread in obj.threads
        thread.task = Threads.@spawn calc_ll_gradient!(thread, obj, feat_dot_parameters, transition_parameters)
    end

    # collect results with early advance
    negll_total = 0.0
    for thread in obj.threads
        wait(thread.task)
        negll_total -= thread.log_likelihood        
        if !isnothing(G)
            # deduce vectorised gradients
            for i in eachindex(thread.state_gradient)
                G[i] -= thread.state_gradient[i]
            end
            for i in eachindex(thread.transition_gradient)
                G[obj.model.state_parameters_count + i] -= thread.transition_gradient[i]
            end
        end
    end

    # exclude the bias feature from being regularized (this assumes it is the very first one!)
    parameters_without_bias = copy(parameters)
    parameters_without_bias[1] = 0

    if obj.model.L1_penalty > 0
        negll_total = regularize_L1(negll_total, G, obj.model.L1_penalty, parameters_without_bias, obj.model.use_L1_clipping)
    end
    if obj.model.L2_penalty > 0
        negll_total = regularize_L2(negll_total, G, obj.model.L2_penalty, parameters_without_bias)
    end

    return negll_total

end

function calc_ll_gradient!(thread, obj, feat_dot_parameters, transition_parameters)

    fill!(thread.state_gradient, 0)
    fill!(thread.transition_gradient, 0)
    thread.log_likelihood = 0.0

    for (idx, x) in enumerate(thread.X)

        forward!(thread.forward_table, thread.transition_table, feat_dot_parameters, x, 
        transition_parameters, obj.model.transitions; need_transition_table = true)

        backward!(thread.backward_table, feat_dot_parameters, x, 
            transition_parameters, obj.model.transitions)

        thread.log_likelihood += log_likelihood!(thread.state_gradient,
                                    thread.transition_gradient,
                                    obj.features, x, thread.y[idx],
                                    obj.model.transitions,
                                    thread.forward_table,
                                    thread.transition_table,
                                    thread.backward_table)
    end

end

function regularize_L1(ll, gradient, c1, parameters, use_clipping::Bool)
    # we have negative ll and gradient here
    ll += c1 * sum(abs, parameters)
    if !isnothing(gradient)
        if use_clipping
            mask = (gradient .< -c1) .| (gradient .> c1)
        end
        gradient .+= c1 * sign.(parameters)
        if use_clipping
            gradient .*= mask
        end
    end
    return ll
end

function regularize_L2(ll, gradient, c2, parameters)
    # we have negative ll and gradient here
    ll += c2 * sum(x-> x^2, parameters)
    if !isnothing(gradient)
        gradient .+= 2.0 * c2 * parameters
    end
    return ll
end

function forward!(forward_table, transition_table, feat_dot_parameters, sample, transition_parameters, transitions; need_transition_table::Bool)

    _, n_states, n_classes = size(forward_table)
    n_time_steps = length(sample)

    # careful: forward_table is pre-allocated per thread for the largest sample, and may contain garbage beyond n_time_steps+1
    fill!( view(forward_table, 1:n_time_steps + 1, 1:n_states, 1:n_classes), -Inf)
    forward_table[1, 1, :] .= 0

    if need_transition_table
        # careful: transition_table is pre-allocated per thread for the largest sample, and may contain garbage beyond n_time_steps
        fill!( view(transition_table, 1:n_time_steps, 1:n_states, 1:n_states, 1:n_classes), -Inf)
    end

    @inbounds for t in 1:n_time_steps, (tridx, tr) in enumerate(transitions)
        class = tr[1]
        s0 = tr[2]
        s1 = tr[3]
        # ft[t, s1] += ft[t-1, s0] * ψ(x_t, s1) * ψ(s0, s1) # ft is +1 length
        # the end term is xdp[s1,class,t] where xdp:=reshape(feat_dot_parameters[:, sample],n_states,n_classes,n_time_steps) - but this avoids copying
        edge_potential = forward_table[t, s0, class] + transition_parameters[tridx] + feat_dot_parameters[s1 + (class-1)*n_states, sample[t]]
        forward_table[t+1, s1, class] =
            logaddexp( forward_table[t+1, s1, class], edge_potential )
        if need_transition_table
            # ftt[t, s0, s1] += ft[t-1, s0] * ψ(x_t, s1) * ψ(s0, s1)
            transition_table[t, s0, s1, class] =
                logaddexp( transition_table[t, s0, s1, class], edge_potential )
        end
    end

end

function backward!(backward_table, feat_dot_parameters, sample, transition_parameters, transitions)

    _, n_states, n_classes = size(backward_table)
    n_time_steps = length(sample)

    # careful: forward_table is pre-allocated per thread for the largest sample, and may contain garbage beyond n_time_steps+1
    fill!( view(backward_table, 1:n_time_steps+1, 1:n_states, 1:n_classes), -Inf)
    backward_table[n_time_steps+1, end, :] .= 0

    @inbounds for t in n_time_steps:-1:1, (tridx, tr) in enumerate(transitions)
        class = tr[1]
        s0 = tr[2]
        s1 = tr[3]
        # bt[t-1, s0] += bt[t, s1] * ψ(x_t, s1) * ψ(s0, s1) # bt is +1 length
        # the end term is xdp[s1,class,t] where xdp:=reshape(feat_dot_parameters[:, sample],n_states,n_classes,n_time_steps) - but this avoids copying
        edge_potential = backward_table[t + 1, s1, class] + transition_parameters[tridx] + feat_dot_parameters[s1 + (class-1)*n_states, sample[t]]
        backward_table[t, s0, class] =
            logaddexp( backward_table[t, s0, class], edge_potential )
    end

end

function log_likelihood!(state_gradient, transition_gradient, features, sample, cy, transitions, 
    forward_table, transition_table, backward_table)

    n_states, n_classes, n_features = size(state_gradient)
    n_time_steps = length(sample)

    # compute Z by rewinding the forward table for all classes
    Z = -Inf
    for c in 1:n_classes
        Z = logaddexp(Z, forward_table[n_time_steps+1, n_states, c])
    end

    if issparse(features)
        x_rows = rowvals(features)
        x_vals = nonzeros(features)
    end

    # state parameter gradient
    @inbounds for t in 1:n_time_steps, state in 1:n_states, c in 1:n_classes
        alphabeta = forward_table[t+1, state, c] + backward_table[t+1, state, c]
        weight = -exp(alphabeta - Z)
        if c == cy
            weight += exp(alphabeta - forward_table[n_time_steps+1, n_states, c])
        end
        if !issparse(features)
            for feat in 1:n_features
                state_gradient[state, c, feat] += weight * features[feat, sample[t]]
            end
        else
            for i in nzrange(features, sample[t])
                state_gradient[state, c, x_rows[i]] += weight * x_vals[i]
            end
        end
    end

    # transition gradient
    @inbounds for t in 1:n_time_steps, (tridx, tr) in enumerate(transitions)
        c = tr[1]
        s0 = tr[2]
        s1 = tr[3]
        alphabeta = transition_table[t, s0, s1, c] + backward_table[t+1, s1, c]
        weight = -exp(alphabeta - Z)
        if c == cy
            weight += exp(alphabeta - forward_table[n_time_steps+1, n_states, c])
        end
        transition_gradient[tridx] += weight
    end
    
    return forward_table[n_time_steps+1, n_states, cy] - Z

end
