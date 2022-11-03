using Optim

struct ConvergenceError <: Exception
    m::HCRFModel
    result::Optim.OptimizationResults
end

"""
fit!(
        X::AbstractVector{<:AbstractArray},
        y::AbstractVector;
        num_states::UInt = 2,
        L1_penalty::Float64 = 0.15,
        L2_penalty::Float64 = 0.15,
        transitions = Nothing,
        state_parameters_noise = 0.001,
        transition_parameter_noise = 0.001,
        use_L1_clipping = false,
        optimize_parameters...)

Fits a HCRF model to the provided features in X and observed classes in y. Length of X and y must match. y may
contain arbitrary values, the dictionary is determined from the range of provided samples.

Returns the fitted model and the optimization result. May throw ConvergenceError with the model and the partial result
if the optimization did not converge.

The X vector stores observation sequences, each containing feature values in a timesteps x features matrix. Timesteps
may differ over sequences, but number of features must match. The very first feature is assumed to be a bias
parameter (constant 1), and its parameter is not included in the L1/L2 regularization.

Arguments:
    X: the samples in a vector of arrays
    y: labels, one for each X
    num_states: number of hidden states
    L1_penalty: L1 penalty multiplier (set to 0 to turn off L1 regularization)
    L2_penalty: L2 penalty multiplier (set to 0 to turn off L2 regularization)
    transitions: contains a list of all possible transitions between hidden states if conditioned on a class label index.
                 Each of these transitions is going to have a weight that can be tuned.
                 The list contains triplets, where the first element is the class label, then hidden state index from, then to.
                 If set to nothing, a default transition table is going to be used that allows sequential change of states.
    state_parameter_noise, transition_parameter_noise: variance for weight initialisation noise
    use_L1_clipping: if true, use clipping of weights in L1 regularisation (gradient-wise dubious)

    Any further parameters will be passed to the Optim.optimize() call. If none, LBFGS is used.
"""
function fit!(X::AbstractVector{<:AbstractArray}, y::AbstractVector;
              num_states::Int = 2,
              L1_penalty::Float64 = 0.15,
              L2_penalty::Float64 = 0.15,
              transitions = nothing,
              state_parameter_noise = 0.001,
              transition_parameter_noise = 0.001,
              use_L1_clipping = false,
              optimize_parameters...)

    if isempty(optimize_parameters)
        optimize_parameters = Dict( :method => Optim.LBFGS() )
    end

    # sanity checks
    @assert length(X) == length(y) "observations and labels do not match in length"
    @assert length(X) > 1 "no observation was provided"
    num_features = size(first(X),2)
    for i in eachindex(X)[2:end]
        @assert size(X[i],2) == num_features "observation $i differ in feature size (was $(size(X[i],2)) while first had $num_features)"
    end

    classes = sort(unique(y))
    num_classes = length(classes)
    classes_map = Dict( cls => i for (i,cls) in enumerate(classes) )

    if isnothing(transitions)
        indexed_transitions = create_default_transitions(num_classes, num_states)
    else
        # convert labels into indices
        indexed_transitions = Vector{Vector{Int}}()
        for t in eachindex(transitions)
            push!(indexed_transitions, [ classes_map[transitions[t][1]], transitions[t][2], transitions[t][3] ] )
        end
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
    obj = ObjectiveFunc(m, X, y)

    result = optimize( Optim.only_fg!(obj), m.parameters; optimize_parameters...)

    copyto!( m.parameters, Optim.minimizer(result) )    

    if !Optim.converged(result)
        throw(ConvergenceError(m, result))
    end

    return m, result

end

# This function generates a default transition table that can only increase in state,
# but can jump from the initial to any, and jump from any to the end state.
# Reindexed to 1-based. The original pyhcrf code looks like buggy, as it contains
# multiple instances of the same transition.
function create_default_transitions(num_classes, num_states)

    transitions = []

    for c in 1:num_classes

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
Predict the class for each sample in X.

Arguments:
    X: a #samples sized vector of #timesteps x #features sized matrices.
        Contains the samples to predict labels for. Samples must have the same
        number of features as the samples used for training.

Returns:
    A #samples sized vector containing the class
    label with the highest probability for each sample in X. Use
    predict_marginals() to access all class probabilities.
"""
function predict(m::HCRFModel, X::AbstractVector{T}) where T <: AbstractArray
    preds = predict_marginals(m, X)
    return [ findmax(pred)[2] for pred in preds ]

end

"""
Estimate all class probabilities for each sample in X.

The returned estimates for all classes are ordered by the
label of classes.

Arguments:
    X: a #samples sized vector of #timesteps x #features sized matrices.
        Contains the samples to predict labels for. Samples must have the same
        number of features as the samples used for training.

Returns:
    A #samples sized vector containing dictionaries mapping class labels
    to the prediction probability for each sample of X.
"""
function predict_marginals(m::HCRFModel, X::AbstractVector{T}) where T <: AbstractArray

    y = []
    for x in X
        n_time_steps, n_features = size(x)
        _, n_states, n_classes = size(m.state_parameters)
        x_dot_parameters = reshape(x * reshape(m.state_parameters, n_features, :), n_time_steps, n_states, n_classes)

        forward_table = forward(x_dot_parameters,
                                m.state_parameters,
                                m.transition_parameters,
                                m.transitions)
        
        norm_preds = exp.( forward_table[end, end, :] .- logsumexp(forward_table[end, end, :]) )

        push!(y, Dict(zip(m.classes, norm_preds)))
    end
    return y
end
