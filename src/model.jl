using Optim

"""
    fit!( X::AbstractVector{<:AbstractArray}, y::AbstractVector;
        observations = nothing,
        num_states::UInt = 3,
        L1_penalty::Float64 = 0.,
        L2_penalty::Float64 = 0.15,
        transition_generator = unconstrained_transitions,
        state_parameters_noise = 0.001,
        transition_parameter_noise = 0.001,
        use_L1_clipping = false,
        optimize_parameters...)

Fits a HCRF model to the provided features in `X` and observed classes in `y`. Length of `X` and `y` must match. `y` may
contain any values that can be used as keys in a `Dict`.

The `X` vector stores observation sequences. Two formats are accepted:

1. a vector of arrays, where each array contains feature values in a `#timesteps x #features` matrix.
2. if the `observations` argument is filled, samples in X are viewed row index lists of that table.

Matrices can be of arbitrary format until they accept matrix multiplication with a full double matrix. Timesteps
may differ but number of features must match over samples. The very first feature is assumed to be a bias
parameter (constant 1), and it is excluded from the L1/L2 regularization.

If samples are provided via the observation list, probabilities are calculated only once for each possible
observation. This can speed-up computations in case when they can take a limited set of values (e.g. overlapping).

# Further arguments:

- `num_states`: number of hidden states, *including* the obligatory start and end states
- `L1_penalty`: L1 penalty multiplier (set to 0 to turn off L1 regularization)
- `L2_penalty`: L2 penalty multiplier (set to 0 to turn off L2 regularization)
- `transition_generator`: a function that accepts a class alphabet and the number of hidden states, then outputs
                a list of all possible transitions between hidden states if conditioned on a class label.
                The list must contain triplets, where the first element is the class label, then hidden state index from, then to.
                Each of these transitions is going to have a weight that can be tuned.
                If not specified, the `unconstrained_transitions()` function is used, that allows arbitrary changes.
                See also `step_transitions()` for another example generator.

- `state_parameter_noise`, `transition_parameter_noise`: variance for weight initialisation noise

- `use_L1_clipping`: if true, use clipping of weights in L1 regularisation (gradient-wise dubious)

Any further arguments not listed above will be passed to the `Optim.optimize()` call. If none such, `LBFGS()` is used.

# Returns:

The fitted model and the optimization result. May throw `ConvergenceError` with the model and the partial result
if the optimization did not converge.

"""
function fit!(X::AbstractVector{<:AbstractArray}, y::AbstractVector;
              observations = nothing,
              num_states::Int = 3,
              L1_penalty::Float64 = 0.,
              L2_penalty::Float64 = 0.15,
              transition_generator::Function = unconstrained_transitions,
              state_parameter_noise = 0.001,
              transition_parameter_noise = 0.001,
              use_L1_clipping = false,
              optimize_parameters...)

    if isempty(optimize_parameters)
        optimize_parameters = Dict( :method => Optim.LBFGS() )
    end

    # sanity checks
    @assert num_states > 1 "num_states must be larger than 1 (init and end states are needed)"
    @assert length(X) == length(y) "observations and labels do not match in length"
    @assert length(X) > 1 "no observation was provided"
    for i in eachindex(X)
        @assert size(X[i],1) > 0 "empty observations were provided at index $(i)"
    end
    num_features = isnothing(observations) ? size(first(X),2) : size(observations,2)
    @assert num_features > 0 "first observation has no features"
    if isnothing(observations)
        for i in eachindex(X)[2:end]
            @assert size(X[i],2) == num_features "observation $i differ in feature size (was $(size(X[i],2)) while first had $num_features)"
        end
    end
 
    classes = sort(unique(y))
    num_classes = length(classes)
    classes_map = Dict( cls => i for (i,cls) in enumerate(classes) )

    transitions = transition_generator( classes, num_states )

    # convert labels into indices
    indexed_transitions = Vector{Vector{Int}}()
    for t in eachindex(transitions)
        @assert transitions[t][1] in keys(classes_map) "found unknown class in transitions: " transitions[t][1]
        push!(indexed_transitions, [ classes_map[transitions[t][1]], transitions[t][2], transitions[t][3] ] )
    end

    num_transitions = length(indexed_transitions)
    state_parameters_shape = (num_features, num_states, num_classes)
    state_parameters_count = prod(state_parameters_shape)

    parameters = zeros(state_parameters_count + num_transitions)

    # views
    state_parameters = reshape( view(parameters, 1:state_parameters_count), state_parameters_shape )
    transition_parameters = view(parameters, state_parameters_count+1:(state_parameters_count + num_transitions) )

    # random initialization of parameters
    s = randn(state_parameters_shape) * state_parameter_noise
    copyto!(state_parameters, s)
    t = randn( size(transition_parameters) ) * transition_parameter_noise
    copyto!(transition_parameters, t)

    m = HCRFModel(num_states, L1_penalty, L2_penalty, use_L1_clipping, state_parameter_noise, transition_parameter_noise,
              indexed_transitions, state_parameters_shape, state_parameters_count, parameters,
              state_parameters, transition_parameters, classes, classes_map)

    # function object closure
    obj = ObjectiveFunc(m, X, y, observations)

    result = optimize( Optim.only_fg!(obj), m.parameters; optimize_parameters...)

    copyto!( m.parameters, Optim.minimizer(result) )    

    if !Optim.converged(result)
        throw(ConvergenceError(m, result))
    end

    return m, result

end

"""
    step_transitions(classes_alphabet::AbstractVector, num_states::Int)

Generates a transition table that can increase or stay in hidden
state number at each step, but can jump from the initial state to any, and jump from
any to the end state. (Fixed from the original pyhcrf version that contains
multiple instances of the same transition.)
"""
function step_transitions(classes_alphabet::AbstractVector, num_states::Int)

    transitions = []

    for c in classes_alphabet

        for state in 1:num_states
            push!(transitions, [c, state, state])  # stay in the same
            if state > 1
                push!(transitions, [c, state-1, state])  # to the next state
            end
            if state > 2
                push!(transitions, [c, 1, state])  # jump from the initial (also to remain there)
            end
            if state > 1 && state < num_states - 1
                push!(transitions, [c, state, num_states]) # to the end state
            end
        end

    end

    return sort(transitions)

end

"""
    unconstrained_transitions(classes_alphabet::AbstractVector, num_states::Int)

Generates a transition table that allows staying or moving to any non-start and
non-end hidden state at each step.

In contrast to `step_transitions()`, the state machine
is going to visit the initial state exactly once at the beginning, and visit 
the end state exactly once as the last.

The start state may continue at any intermediate
state, and the end state can be reached from any intermediate state, therefore the intermediate
states are unconstrained in start and ending.
"""
function unconstrained_transitions(classes_alphabet::AbstractVector, num_states::Int)

    transitions = []

    for c in classes_alphabet

        # from-to any state, including from start and to end, and also itself
        # forbid self transition for start and end states
        for state_from in 1:num_states-1
            for state_to in 2:num_states
                push!(transitions, [c, state_from, state_to])
            end
        end
    end

    return sort(transitions)

end

"""
    predict(m::HCRFModel, X::AbstractVector{<:AbstractArray}; observations = nothing)

Predict the class by the fitted model in `m` for each sample in `X`.

If `observations` is provided, `X` is assumed to contain list of row indices into this matrix (see `fit!()`).

Samples must have the same number of features as the samples used for training.

# Returns:

A `#samples` sized vector containing the class labels with the highest probability
for each sample in `X`. Use `predict_marginals()` to access more details.
"""
function predict(m::HCRFModel, X::AbstractVector{<:AbstractArray}; observations = nothing)
    preds, _ = predict_marginals(m, X; observations)
    return [ findmax(pred)[2] for pred in preds ]

end

"""
    predict_marginals(m::HCRFModel, X::AbstractVector{<:AbstractArray}; observations = nothing, calc_hidden = false)

Estimate all class probabilities for each sample in `X`, and optionally calculate
the most probable hidden state sequence.

# Arguments:

- `m`: the fitted model
- `X`, `observations`: samples to predict labels for (see `fit!()`).
- `calc_hidden`: if true, calculate and return the most likely hidden state sequence in the second return value.

# Returns:

1. a `#samples` sized vector containing dictionaries mapping class labels to the 
    prediction probability for each sample of `X`.
2. nothing if `calc_hidden` was false, otherwise the sequence of predicted hidden state
        labels for each `X`. It always starts in state 1, and ends in `#states`, but it
        is going to be only one timestep longer than the original sample (but uses all `X` data).
"""
function predict_marginals(m::HCRFModel, X::AbstractVector{<:AbstractArray}; observations = nothing, calc_hidden = false)

    y = []

    n_features, n_states, n_classes = size(m.state_parameters)

    if !calc_hidden
        hidden_states = nothing
    else
        hidden_states = []
    end

    # sanity checks
    @assert length(X) > 1 "no observation was provided"
    for i in eachindex(X)
        @assert size(X[i],1) > 0 "empty observations were provided at index $(i)"
    end
    if isnothing(observations)
        for i in eachindex(X)
            @assert size(X[i],2) == n_features "observation $i differ in feature size (was $(size(X[i],2)) while it was $n_features at learning)"
        end
    else
        @assert size(observations, 2) == n_features "observations differ in feature size ($(size(observations,2)) while it was $n_features at learning)"

        # create cache
        obs_dot_parameters = observations * reshape(m.state_parameters, n_features, :)
    end

    for x in X
        n_time_steps = size(x, 1)
        if isnothing(observations)
            x_dot_parameters = reshape(x * reshape(m.state_parameters, n_features, :), n_time_steps, n_states, n_classes)
        else
            x_dot_parameters = reshape(view(obs_dot_parameters, x, :), n_time_steps, n_states, n_classes)
        end

        forward_table, transition_table, _ = forward_backward(x_dot_parameters,
                                m.state_parameters,
                                m.transition_parameters,
                                m.transitions;
                                need_transition_table = calc_hidden,
                                need_backward_table = false)
        
        norm_preds = exp.( forward_table[end, end, :] .- logsumexp(forward_table[end, end, :]) )

        push!(y, Dict(zip(m.classes, norm_preds)))

        if calc_hidden

            maxy = argmax( forward_table[end, end, :] )
            # start from end state
            act_hidden_states = [ size(transition_table,2) ]

            for t in reverse(axes(transition_table,1))
                push!(act_hidden_states, argmax( transition_table[t, :, act_hidden_states[end], maxy] ) )
            end 

            push!(hidden_states, reverse!(act_hidden_states))

        end
    end
    return y, hidden_states
end
