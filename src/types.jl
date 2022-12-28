mutable struct HCRFModel
    # parameters used in the fitmodel!() that produced this model
    num_states::UInt
    L1_penalty::Float64 # L1 penalty constant used in the fit
    L2_penalty::Float64 # L2 penalty constant used in the fit
    use_L1_clipping::Bool # if false, no clipping was used in L1
    state_parameter_noise::Float64 # variance of init normal noise
    transition_parameter_noise::Float64 # variance of init normal noise
    transitions::Vector{Vector{Int64}}

    state_parameters_shape::Tuple{Int64, Int64, Int64}
    state_parameters_count::Int64

    # theta and its views
    parameters::Vector{Float64}
    state_parameters::Base.ReshapedArray
    transition_parameters::SubArray

    # the dictionary of classes
    # i-th class is associated with label 'i' internally
    classes::AbstractVector
    classes_map::Dict
end

# accumulated values per thread
mutable struct HCRFThread
    task::Task
    X::Vector{Vector{Int64}}
    y::Vector{Int64}
    state_gradient::Array{Float64,3}
    transition_gradient::Vector{Float64}
    forward_table::Array{Float64,3}
    transition_table::Array{Float64,4}
    backward_table::Array{Float64,3}
    log_likelihood::Float64
end

struct ObjectiveFunc{T <: AbstractArray}
    model::HCRFModel
    features::T
    threads::Vector{HCRFThread}

    function ObjectiveFunc(model, X, y, features, num_threads)

        n_states, n_classes, _ = model.state_parameters_shape

        # to avoid excessive allocation of memory, we
        # store only one instance of some temporary variables per thread.
        # We need to determine the longest sample length to pre-allocate.
        # Care should be taken however to use the forward/backward/transition
        # tables up to the actual sample length only, as beyond they may contain
        # garbage over subsequent samples.

        max_time_steps = maximum( length.(X) )

        X_chunks = equal_partition(X, num_threads)
        y_chunks = equal_partition(y, num_threads)

        # use less threads if more than samples
        num_threads = min(length(X_chunks), num_threads)

        threads = HCRFThread[]
        for t in 1:num_threads
            push!( threads, HCRFThread( Task(""), 
                copy(X_chunks[t]), copy(y_chunks[t]),
                Array{Float64}(undef, model.state_parameters_shape),
                Vector{Float64}(undef, length(model.transitions) ),
                Array{Float64}(undef, max_time_steps + 1, n_states, n_classes),
                Array{Float64}(undef, max_time_steps, n_states, n_states, n_classes),
                Array{Float64}(undef, max_time_steps + 1, n_states, n_classes),
                0 ) )
            end
    
        new{typeof(features)}(model, features, threads)
    
    end

end
