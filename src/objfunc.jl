# evaluate the objective function (called by Optim.jl)
function (obj::ObjectiveFunc)(_, G, parameters::Vector{Float64})

    negll_total = 0.0

    if !isnothing(G)
        fill!(G, 0)
    end

    n_states, n_classes, n_features = obj.model.state_parameters_shape

    state_parameters = reshape( view(parameters, 1:obj.model.state_parameters_count), obj.model.state_parameters_shape )
    transition_parameters = view(parameters, obj.model.state_parameters_count+1:(obj.model.state_parameters_count + length(obj.model.transitions) ) )

    feat_dot_parameters = reshape(state_parameters, :, n_features) * obj.features

    for (idx, sample, class) in zip(1:length(obj.X), obj.X, obj.y)

        n_time_steps = length(sample)

        # rowvals() does not work on a sparse view with arbitrary cols, hence the copy
        # x = copy(view(obj.features, :, sample))
        x = obj.features[:, sample]

        # x_dot_parameters = copy( reshape( view(feat_dotparameters, :, sample), n_states, n_classes, n_time_steps))
        # x_dot_parameters = reshape( view(feat_dot_parameters, :, sample), n_states, n_classes, n_time_steps)
        x_dot_parameters = reshape( feat_dot_parameters[:, sample], n_states, n_classes, n_time_steps)

        forward!(obj.forward_tables[idx], obj.transition_tables[idx], x_dot_parameters,
                 transition_parameters, obj.model.transitions; need_transition_table = true)

        backward!(obj.backward_tables[idx], x_dot_parameters,
                  transition_parameters, obj.model.transitions)

        negll_total -= log_likelihood!(obj.state_gradients[idx],
                                       obj.transition_gradients[idx],
                                       x, class,
                                       obj.model.transitions,
                                       obj.forward_tables[idx],
                                       obj.transition_tables[idx],
                                       obj.backward_tables[idx])
                   
    end

    # deduce vectorised gradients
    if !isnothing(G)
        for idx in 1:length(obj.X)
            for i in eachindex(obj.state_gradients[idx])
                G[i] -= obj.state_gradients[idx][i]
            end
            for i in eachindex(obj.transition_gradients[idx])
                G[obj.model.state_parameters_count + i] -= obj.transition_gradients[idx][i]
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

function regularize_L1(ll, gradient, c1, parameters, use_clipping::Bool)
    # compared to Python, we have negative ll and gradient here
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
    # compared to Python, we have negative ll and gradient here
    ll += c2 * sum(x-> x^2, parameters)
    if !isnothing(gradient)
        gradient .+= 2.0 * c2 * parameters
    end
    return ll
end

function forward!(forward_table, transition_table, x_dot_parameters, transition_parameters, transitions; need_transition_table::Bool)

    n_time_steps = size(x_dot_parameters, 3)

    fill!(forward_table, -Inf)
    forward_table[1, 1, :] .= 0

    if need_transition_table
        transition_table = fill!( transition_table, -Inf)
    end

    @inbounds for t in 1:n_time_steps, (tridx, tr) in enumerate(transitions)
        class = tr[1]
        s0 = tr[2]
        s1 = tr[3]
        # ft[t, s1] += ft[t-1, s0] * ψ(x_t, s1) * ψ(s0, s1) # ft is +1 length
        edge_potential = forward_table[t, s0, class] + transition_parameters[tridx] + x_dot_parameters[s1, class, t]
        forward_table[t+1, s1, class] =
            logaddexp( forward_table[t+1, s1, class], edge_potential )
        if need_transition_table
            # ftt[t, s0, s1] += ft[t-1, s0] * ψ(x_t, s1) * ψ(s0, s1)
            transition_table[t, s0, s1, class] =
                logaddexp( transition_table[t, s0, s1, class], edge_potential )
        end
    end

end

function backward!(backward_table, x_dot_parameters, transition_parameters, transitions)

    n_states = size(x_dot_parameters, 1)
    n_time_steps = size(x_dot_parameters, 3)

    fill!(backward_table, -Inf)
    backward_table[n_time_steps + 1, n_states, :] .= 0

    @inbounds for t in n_time_steps:-1:1, (tridx, tr) in enumerate(transitions)
        class = tr[1]
        s0 = tr[2]
        s1 = tr[3]
        # bt[t-1, s0] += bt[t, s1] * ψ(x_t, s1) * ψ(s0, s1) # bt is +1 length
        edge_potential = backward_table[t + 1, s1, class] + x_dot_parameters[s1, class, t] + transition_parameters[tridx]
        backward_table[t, s0, class] =
            logaddexp( backward_table[t, s0, class], edge_potential )
    end

end

function log_likelihood!(state_gradient, transition_gradient, x, cy, transitions, 
    forward_table, transition_table, backward_table)

    n_states, n_classes, n_features = size(state_gradient)
    n_time_steps = size(x, 2)

    # reset parameter gradients buffers
    fill!(state_gradient, 0)
    fill!(transition_gradient, 0)

    # compute Z by rewinding the forward table for all classes
    Z = -Inf
    for c in 1:n_classes
        Z = logaddexp(Z, forward_table[n_time_steps+1, n_states, c])
    end

    if issparse(x)
        x_rows = rowvals(x)
        x_vals = nonzeros(x)
    end

    # compute all state parameter gradients
    @inbounds for t in 1:n_time_steps, state in 1:n_states, c in 1:n_classes
        alphabeta = forward_table[t+1, state, c] + backward_table[t+1, state, c]
        weight = -exp(alphabeta - Z)
        if c == cy
            weight += exp(alphabeta - forward_table[n_time_steps+1, n_states, c])
        end
        if !issparse(x)
            for feat in 1:n_features
                state_gradient[state, c, feat] += weight * x[feat, t]
            end
        else
            for i in nzrange(x, t)
                state_gradient[state, c, x_rows[i]] += weight * x_vals[i]
            end
        end
    end

    # compute all transition parameter gradients
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
