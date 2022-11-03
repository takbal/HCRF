# evaluate the objective function (called by Optim.jl)
function (obj::ObjectiveFunc)(_, G, x)

    negll_total = 0.0

    if !isnothing(G)
        fill!(G, 0)
    end

    state_parameters = reshape( view(x, 1:obj.model.state_parameters_count), obj.model.state_parameters_shape ) # features x states x classes sized 3D
    transition_parameters = view(x, obj.model.state_parameters_count+1:(obj.model.state_parameters_count + length(obj.model.transitions) ) )

    # destination vector to fill (presented by views)
    # consider: filling G directly if log_likelihood can deduct it there (but that makes parallelism hard)
    gradient = similar(obj.model.parameters)
    state_gradient = reshape( view(gradient, 1:obj.model.state_parameters_count), obj.model.state_parameters_shape ) # features x states x classes sized 3D
    transition_gradient = view(gradient, obj.model.state_parameters_count+1:(obj.model.state_parameters_count + length(obj.model.transitions) ) )

    for (sample, class) in zip(obj.X, obj.y)
        negll_total -= log_likelihood(sample, obj.model.classes_map[class],
                             state_parameters,
                             transition_parameters,
                             obj.model.transitions,
                             state_gradient,
                             transition_gradient)
                         
        if !isnothing(G)
            G .-= gradient
        end
    end
  
    # exclude the bias feature from being regularized (this assumes it is the very first one!)
    parameters_without_bias = copy(x)
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

# numerically stable log( exp(x) + exp(y) )
function logaddexp(x::Number, y::Number)
    if x == y
        # infs
        return x + log(2)
    else
        tmp = x - y
        if tmp > 0
            return x + log1p(exp(-tmp))
        elseif tmp <= 0
            return y + log1p(exp(tmp))
        else # nans
            return tmp
        end
    end
end

# numerically stable Σ log( exp(x_i) )
function logsumexp(x::AbstractArray{I}) where {I<:Number}
    m = maximum(x)
    r = 0.0
    for i in eachindex(x)
        r += exp(x[i] - m)
    end
    return log(r) + m
end

function forward(x_dot_parameters, state_parameters, transition_parameters, transitions)

    n_time_steps = size(x_dot_parameters, 1)
    n_states = size(state_parameters, 2)
    n_classes = size(state_parameters, 3)

    # it stores probabilities in log space
    # Add extra 1 time step for start state
    forward_table = fill!( Array{Float64}(undef, n_time_steps + 1, n_states, n_classes), -Inf)
    forward_table[1, 1, :] .= 0

    for t in 2:n_time_steps+1
        for (tridx, tr) in enumerate(transitions)
            class_number = tr[1]
            s0 = tr[2]
            s1 = tr[3]
            # all this does is ft[t, s1] += ft[t-1, s0] + theta * features[t, s1] (features are in unextended t space)
            edge_potential = forward_table[t - 1, s0, class_number] + transition_parameters[tridx]
            forward_table[t, s1, class_number] = logaddexp(
                forward_table[t, s1, class_number],
                edge_potential + x_dot_parameters[t - 1, s1, class_number]
            )
        end
    end

    return forward_table

end

function forward_backward(x_dot_parameters, state_parameters, transition_parameters, transitions)

    n_time_steps = size(x_dot_parameters, 1)
    n_states = size(state_parameters, 2)
    n_classes = size(state_parameters, 3)
    n_transitions = length(transitions)

    # it stores probabilities in log space
    # Add extra 1 time step for start state
    forward_table = fill!( Array{Float64}(undef, n_time_steps + 1, n_states, n_classes), -Inf)
    forward_table[1, 1, :] .= 0

    forward_transition_table = fill!( Array{Float64}(undef, n_time_steps + 1, n_states, n_states, n_classes), -Inf)

    backward_table = fill!( Array{Float64}(undef, n_time_steps + 1, n_states, n_classes), -Inf)
    backward_table[n_time_steps + 1, n_states, :] .= 0

    # Compute forward transitions
    for t in 2:n_time_steps+1
        for (tridx, tr) in enumerate(transitions)
            class_number = tr[1]
            s0 = tr[2]
            s1 = tr[3]
            # all this does is ft[t, s1] += ft[t-1, s0] + theta * features[t, s1] (features are in unextended t space)
            edge_potential = forward_table[t - 1, s0, class_number] + transition_parameters[tridx]
            forward_table[t, s1, class_number] = logaddexp(
                forward_table[t, s1, class_number],
                edge_potential + x_dot_parameters[t - 1, s1, class_number]
            )
            # all this does is ftt[t, s0, s1] += ft[t-1, s0] + theta * features[t, s1] (features are in unextended t space)
            forward_transition_table[t, s0, s1, class_number] = logaddexp(
                forward_transition_table[t, s0, s1, class_number],
                edge_potential + x_dot_parameters[t - 1, s1, class_number]
            )
        end
    end
    # Compute backwards transitions
    for t in n_time_steps:-1:1
        for (tridx, tr) in enumerate(transitions)
            class_number = tr[1]
            s0 = tr[2]
            s1 = tr[3]
            # all this does is bt[t, s0] += bt[t+1, s1] + theta * features[t+1, s1] (features are in unextended t space)
            edge_potential = backward_table[t + 1, s1, class_number] + x_dot_parameters[t, s1, class_number]
            backward_table[t, s0, class_number] = logaddexp(
                backward_table[t, s0, class_number],
                edge_potential + transition_parameters[tridx]
            )
        end
    end

    return forward_table, forward_transition_table, backward_table

end

function log_likelihood(x, cy, state_parameters, transition_parameters,
    transitions, dstate_parameters, dtransition_parameters)

    n_time_steps = size(x, 1)
    n_features = size(x, 2)
    n_states = size(state_parameters, 2)
    n_classes = size(state_parameters, 3)

    x_dot_parameters = reshape(x * reshape(state_parameters, n_features, :), n_time_steps, n_states, n_classes)

    forward_table, forward_transition_table, backward_table = forward_backward(
        x_dot_parameters,
        state_parameters,
        transition_parameters,
        transitions
    )

    # reset parameter gradients buffers
    fill!(dstate_parameters, 0)
    fill!(dtransition_parameters, 0)

    # compute Z by rewinding the forward table for all classes
    Z = -Inf
    for c in 1:n_classes
        Z = logaddexp(Z, forward_table[n_time_steps+1, n_states, c])
    end

    # compute all state parameter gradients
    for t in 2:n_time_steps+1
        for state in 1:n_states
            for c in 1:n_classes
                alphabeta = forward_table[t, state, c] + backward_table[t, state, c]
                weight = exp(alphabeta - forward_table[n_time_steps+1, n_states, c]) * (c == cy) - exp(alphabeta - Z)
                for feat in 1:n_features
                    dstate_parameters[feat, state, c] += weight * x[t - 1, feat]
                end
            end
        end
    end

    # compute all transition parameter gradients
    for t in 2:n_time_steps+1
        for (tridx, tr) in enumerate(transitions)
            c = tr[1]
            s0 = tr[2]
            s1 = tr[3]
            alphabeta = forward_transition_table[t, s0, s1, c] + backward_table[t, s1, c]
            weight = exp(alphabeta - forward_table[n_time_steps+1, n_states, c]) * (c == cy) - exp(alphabeta - Z)
            dtransition_parameters[tridx] += weight
        end
    end
    
    return forward_table[n_time_steps + 1, n_states, cy] - Z

end
