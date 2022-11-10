# evaluate the objective function (called by Optim.jl)
function (obj::ObjectiveFunc)(_, G, parameters)

    negll_total = 0.0

    if !isnothing(G)
        fill!(G, 0)
    end

    state_parameters = reshape( view(parameters, 1:obj.model.state_parameters_count), obj.model.state_parameters_shape ) # features x states x classes sized 3D
    transition_parameters = view(parameters, obj.model.state_parameters_count+1:(obj.model.state_parameters_count + length(obj.model.transitions) ) )

    # destination vector to fill (presented by views)
    # consider: filling G directly if log_likelihood can deduct it there (but that makes parallelism hard)
    gradient = similar(obj.model.parameters)
    state_gradient = reshape( view(gradient, 1:obj.model.state_parameters_count), obj.model.state_parameters_shape ) # features x states x classes sized 3D
    transition_gradient = view(gradient, obj.model.state_parameters_count+1:(obj.model.state_parameters_count + length(obj.model.transitions) ) )

    n_features, n_states, n_classes = size(state_parameters)

    if !isnothing(obj.observations)
        # create observation cache
        obs_dot_parameters = obj.observations * reshape(state_parameters, n_features, :)
    end

    for (sample, class) in zip(obj.X, obj.y)

        if isnothing(obj.observations)
            # plain matrices
            x = sample
            x_dot_parameters = nothing            
        else
            # observation-indexed vectors
            x = view(obj.observations, sample, :)
            x_dot_parameters = reshape( view(obs_dot_parameters, sample, :), size(sample, 1), n_states, n_classes)
        end
    
        negll_total -= log_likelihood(x, obj.model.classes_map[class],
                             state_parameters,
                             transition_parameters,
                             obj.model.transitions,
                             state_gradient,
                             transition_gradient;
                             x_dot_parameters)
                         
        if !isnothing(G)
            G .-= gradient
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

function forward_backward(x_dot_parameters, state_parameters, transition_parameters, transitions; need_transition_table::Bool, need_backward_table::Bool)

    n_time_steps = size(x_dot_parameters, 1)
    n_states = size(state_parameters, 2)
    n_classes = size(state_parameters, 3)

    # all probabilities are in log space
    # Add extra 1 time step for start state
    forward_table = fill!( Array{Float64}(undef, n_time_steps + 1, n_states, n_classes), -Inf)
    forward_table[1, 1, :] .= 0

    if need_transition_table || need_backward_table
        transition_table = fill!( Array{Float64}(undef, n_time_steps, n_states, n_states, n_classes), -Inf)
    else
        transition_table = nothing
    end

    if need_backward_table
        backward_table = fill!( Array{Float64}(undef, n_time_steps + 1, n_states, n_classes), -Inf)
        backward_table[n_time_steps + 1, n_states, :] .= 0
    else
        backward_table = nothing
    end

    # forward transitions
    for t in 1:n_time_steps
        for (tridx, tr) in enumerate(transitions)
            class_number = tr[1]
            s0 = tr[2]
            s1 = tr[3]
            # ft[t, s1] += ft[t-1, s0] * ψ(x_t, s1) * ψ(s0, s1) # ft is +1 length
            edge_potential = forward_table[t, s0, class_number] + transition_parameters[tridx] + x_dot_parameters[t, s1, class_number]
            forward_table[t+1, s1, class_number] =
                logaddexp( forward_table[t+1, s1, class_number], edge_potential )
            if need_transition_table
                # ftt[t, s0, s1] += ft[t-1, s0] * ψ(x_t, s1) * ψ(s0, s1)
                transition_table[t, s0, s1, class_number] =
                    logaddexp( transition_table[t, s0, s1, class_number], edge_potential )
            end
        end
    end

    if need_backward_table
        # backwards transitions
        for t in n_time_steps:-1:1
            for (tridx, tr) in enumerate(transitions)
                class_number = tr[1]
                s0 = tr[2]
                s1 = tr[3]
                # bt[t-1, s0] += bt[t, s1] * ψ(x_t, s1) * ψ(s0, s1) # bt is +1 length
                edge_potential = backward_table[t + 1, s1, class_number] + x_dot_parameters[t, s1, class_number] + transition_parameters[tridx]
                backward_table[t, s0, class_number] =
                    logaddexp( backward_table[t, s0, class_number], edge_potential )
            end
        end
    end

    return forward_table, transition_table, backward_table

end

function log_likelihood(x, cy, state_parameters, transition_parameters,
    transitions, state_gradient, transition_gradient; x_dot_parameters = nothing)

    n_time_steps = size(x, 1)
    n_features, n_states, n_classes = size(state_parameters)

    if isnothing(x_dot_parameters)
        # allow providing it from the cache
        x_dot_parameters = reshape(x * reshape(state_parameters, n_features, :), n_time_steps, n_states, n_classes)
    end

    forward_table, transition_table, backward_table = forward_backward(
        x_dot_parameters,
        state_parameters,
        transition_parameters,
        transitions;
        need_transition_table = true,
        need_backward_table = true
    )

    # reset parameter gradients buffers
    fill!(state_gradient, 0)
    fill!(transition_gradient, 0)

    final_ft = view(forward_table, n_time_steps+1, n_states, :)

    # compute Z by rewinding the forward table for all classes
    Z = -Inf
    for c in 1:n_classes
        Z = logaddexp(Z, final_ft[c])
    end

    # compute all state parameter gradients
    for t in 1:n_time_steps
        for state in 1:n_states
            for c in 1:n_classes
                alphabeta = forward_table[t+1, state, c] + backward_table[t+1, state, c]
                weight = -exp(alphabeta - Z)
                if c == cy
                    weight += exp(alphabeta - final_ft[c])
                end
                for feat in 1:n_features
                    state_gradient[feat, state, c] += weight * x[t, feat]
                end
            end
        end
    end

    # compute all transition parameter gradients
    for t in 1:n_time_steps
        for (tridx, tr) in enumerate(transitions)
            c = tr[1]
            s0 = tr[2]
            s1 = tr[3]
            alphabeta = transition_table[t, s0, s1, c] + backward_table[t+1, s1, c]
            weight = -exp(alphabeta - Z)
            if c == cy
                weight += exp(alphabeta - final_ft[c])
            end
            transition_gradient[tridx] += weight
        end
    end
    
    return final_ft[cy] - Z

end
