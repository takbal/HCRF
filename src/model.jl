"""
    fit( X::AbstractVector{<:AbstractArray}, y::AbstractVector;
        features = nothing,
        num_states::UInt = 3,
        L1_penalty::Float64 = 0.,
        L2_penalty::Float64 = 0.15,
        transition_generator = unconstrained_transitions,
        state_parameters_noise = 0.001,
        transition_parameter_noise = 0.001,
        use_L1_clipping = false,
        suppress_warning = false,
        callback = nothing,
        one_task_per_sample = false,
        optimize_parameters...)

Fits a HCRF model to the provided features in `X` and observed classes in `y`. Length of `X` and `y` must match. `y` may
contain any values that can be used as keys in a `Dict`.

The `X` vector stores observation sequences. Two formats are accepted:

1. it can be a vector of arrays, where each array contains feature values in a `#features x #timesteps` matrix.
(It is internally converted into the next format.)

2. if the `features` argument is filled, X[i] is considered as a #timesteps long column index list into that table,
and sample matrices are formed from the columns of that table as: `sample_features[i] = features[:, X[i]]`.

Matrices can be of arbitrary format, until they:

1. can be horizontally concatenated into another matrix,
2. after concatenation, they accept multiplication with a full double matrix, that results in a full double matrix,
3. can be indexed by columns.

In particular, sparse feature matrices have a special, faster execution path.

Timesteps may differ, but number of features must match over samples.

The very first feature is assumed to be a bias parameter (a constant 1), and it is excluded from the L1/L2 regularization.

If samples are provided through the `features` parameter, probabilities are calculated only once for each possible
observation. This can speed-up computations if the samples contain overlapping sequences, and the number of features is very large.

# Further arguments:

- `model`: if nothing, generate new model, otherwise use this model for a start. Only the penalty parameters can be changed.
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
- `suppress_warning`: if true, do not print warning messages (e.g. about failed convergence)
- `callback`: a function name or nothing. If specified, call this function at each trace (set frequency by show_every).
              This works the same as Optim.optimize() callbacks, except for it does not only get an Optim.OptimizationState,
              but also the model's actual state as the second parameter. This enables running tests with the actual model during
              optimization, and stop by returning true from the callback. If this is set, extended_trace should be set to true
              for this to work (which is done automatically unless overwritten), and the LBFGS method should be used.
- `num_threads` : number of threads to use. Defaults to Threads.nthreads().

Any further arguments not listed above will be passed to the `Optim.optimize()` call (except for `callback` that is
modified as described above). If none such, `method = LBFGS()` is used.

# Returns:

The fitted model and the optimization result. It may be partial if convergence is not achieved, when @warn is given.

"""
function fit(X::AbstractVector{<:AbstractArray}, y::AbstractVector;
              features = nothing,
              model = nothing,
              num_states::Int = 3,
              L1_penalty::Float64 = 0.,
              L2_penalty::Float64 = 0.15,
              transition_generator::Function = unconstrained_transitions,
              state_parameter_noise = 0.001,
              transition_parameter_noise = 0.001,
              use_L1_clipping = false,
              suppress_warning = false,
              callback = nothing,
              num_threads::Int64 = Threads.nthreads(),
              optimize_parameters...)

    if isempty(optimize_parameters)
        optimize_parameters = Dict( :method => Optim.LBFGS() )
    end

    # sanity checks
    @assert num_states > 1 "num_states must be larger than 1 (init and end states are needed)"
    @assert length(X) == length(y) "samples and labels do not match in length"
    @assert length(X) > 0 "no observation was provided"
    for i in eachindex(X)
        @assert all(size(X[i]) .> 0) "empty samples were provided at index $(i)"
    end
    num_features = isnothing(features) ? size(first(X),1) : size(features,1)
    @assert num_features > 0 "first observation has no features"
    if isnothing(features)
        for i in eachindex(X)[2:end]
            @assert size(X[i],1) == num_features "sample $i differ in feature size (was $(size(X[i],1)) while first had $num_features)"
        end
        # convert X into features table
        X, features = convert_features(X)
    else
        for i in eachindex(X)
            @assert maximum(X[i]) <= size(features,2) "too large index at sample $i"
            @assert minimum(X[i]) >= 1 "too small index at sample $i"
        end
    end

    # process labels
    classes = sort(unique(y))
    num_classes = length(classes)
    classes_map = Dict( cls => i for (i,cls) in enumerate(classes) )
    y_indexed = [ classes_map[value] for value in y ]

    # process transitions
    transitions = transition_generator( classes, num_states )

    # convert labels into indices
    indexed_transitions = Vector{Vector{Int64}}()
    for t in eachindex(transitions)
        @assert transitions[t][1] in keys(classes_map) "found unknown class in transitions: " transitions[t][1]
        push!(indexed_transitions, [ classes_map[transitions[t][1]], transitions[t][2], transitions[t][3] ] )
    end

    if isnothing(model)

        num_transitions = length(indexed_transitions)
        state_parameters_shape = (num_states, num_classes, num_features)
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

    else
        m = deepcopy(model)

        # views
        m.state_parameters = reshape( view(m.parameters, 1:m.state_parameters_count), m.state_parameters_shape )
        m.transition_parameters = view(m.parameters, m.state_parameters_count+1:(m.state_parameters_count + length(m.transitions)) )

        if !suppress_warning && num_states != m.state_parameters_shape[1]
            @warn "specified number of states is different to the model, ignoring"
        end
        @assert m.classes_map == classes_map "classes are different in the model"
        @assert m.state_parameters_shape[3] == num_features "number of features is different in the model"
        if !suppress_warning && indexed_transitions != m.transitions
            @warn "specified transitions are different to the model, ignoring"
        end
        if !suppress_warning && L1_penalty != m.L1_penalty
            @warn "changing L1_penalty to $(L1_penalty) from $(m.L1_penalty)"
            m.L1_penalty = L1_penalty
        end
        if !suppress_warning && L2_penalty != m.L2_penalty
            @warn "changing L2_penalty to $(L2_penalty) from $(m.L2_penalty)"
            m.L2_penalty = L2_penalty
        end
        if !suppress_warning && use_L1_clipping != m.use_L1_clipping
            @warn "changing use_L1_clipping to $(use_L1_clipping) from $(m.use_L1_clipping)"
            m.use_L1_clipping = use_L1_clipping
        end
            
    end

    # function object closure
    obj = ObjectiveFunc(m, X, y_indexed, features, num_threads)

    if !isnothing(callback)
        extended_trace = true
        cb = optstate -> hcrf_callback(optstate; model=m, cb=callback)
    else
        extended_trace = false
        cb = nothing    
    end

    result = optimize( Optim.only_fg!(obj), m.parameters; extended_trace, callback=cb, optimize_parameters...)

    copyto!( m.parameters, Optim.minimizer(result) )    

    if !suppress_warning && !Optim.converged(result)
        @warn "HCRF.fit() failed to converge"
    end

    return m, result

end

function hcrf_callback(optstate; model, cb)
    @assert "x" in keys(optstate.metadata) "you need method=LBFGS() and extended_trace=true for this to work"
    copyto!( model.parameters, optstate.metadata["x"] )
    return cb(optstate, model)
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
    predict(m::HCRFModel, X::AbstractVector{<:AbstractArray}; features = nothing)

Predict the class by the fitted model in `m` for each sample in `X`.

If `features` is provided, `X` is assumed to contain list of row indices into this matrix (see `HCRF.fit()`).

Samples must have the same number of features as the samples used for training.

# Returns:

A `#samples` sized vector containing the class labels with the highest probability
for each sample in `X`. Use `predict_marginals()` to access more details.
"""
function predict(m::HCRFModel, X::AbstractVector{<:AbstractArray}; features = nothing)
    preds, _ = predict_marginals(m, X; features)
    return [ findmax(pred)[2] for pred in preds ]

end

"""
    predict_marginals(m::HCRFModel, X::AbstractVector{<:AbstractArray}; features = nothing, calc_hidden = false)

Estimate all class probabilities for each sample in `X`, and optionally calculate
the most probable hidden state sequence.

# Arguments:

- `m`: the fitted model
- `X`, `features`: samples to predict labels for (see `HCRF.fit()`).
- `calc_hidden`: if true, calculate and return the most likely hidden state sequence in the second return value.

# Returns:

1. a `#samples` sized vector containing dictionaries mapping class labels to the 
    prediction probability for each sample of `X`.
2. nothing if `calc_hidden` was false, otherwise the sequence of predicted hidden state
        labels for each `X`. It always starts in state 1, and ends in `#states`, but it
        is going to be only one timestep longer than the original sample (but uses all `X` data).
"""
function predict_marginals(m::HCRFModel, X::AbstractVector{<:AbstractArray}; features = nothing, calc_hidden = false)

    y = []

    n_states, n_classes, n_features = size(m.state_parameters)

    if !calc_hidden
        hidden_states = nothing
    else
        hidden_states = []
    end

    # sanity checks
    @assert length(X) > 1 "no observation was provided"
    for i in eachindex(X)
        @assert all(size(X[i]) .> 0) "empty sample was provided at index $(i)"
    end

    if isnothing(features)
        for i in eachindex(X)
            @assert size(X[i],1) == n_features "sample $i differ in feature size (was $(size(X[i],1)) while it was $n_features at learning)"
        end

        # convert X into features table
        X, features = convert_features(X)

    else
        @assert size(features, 1) == n_features "features differ in size ($(size(features,1)) while it was $n_features at learning)"
        for i in eachindex(X)
            @assert maximum(X[i]) <= size(features,2) "too large feature index at sample $i"
            @assert minimum(X[i]) >= 1 "too small index at sample $i"
        end
    end

    # create cache
    feat_dot_parameters = reshape(m.state_parameters, :, n_features) * features

    for x in X
        n_time_steps = length(x)

        forward_table = Array{Float64}(undef, n_time_steps + 1, n_states, n_classes)
        if calc_hidden
            transition_table = Array{Float64}(undef, n_time_steps, n_states, n_states, n_classes)
            forward!(forward_table, transition_table, feat_dot_parameters, x,
                     m.transition_parameters, m.transitions; need_transition_table = true)
        else        
            forward!(forward_table, nothing, feat_dot_parameters, x,
                     m.transition_parameters, m.transitions; need_transition_table = false)
        end
        
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

function convert_features(X)
    out = Vector{Vector{Int64}}()
    startidx = 1
    for idx in eachindex(X)
        push!(out, startidx:(startidx + size(X[idx],2) - 1) )
        startidx = startidx + size(X[idx],2)
    end
    features = hcat(X...)
    return out, features
end