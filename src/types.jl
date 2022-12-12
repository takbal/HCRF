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

mutable struct Sample
    x::Vector{Int64}
    y::Int64
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
    samples::Vector{Sample}
    tasks::Vector{Task}
    one_task_per_sample::Bool

    function ObjectiveFunc(model, X, y, features, one_task_per_sample)

        n_states, n_classes, _ = model.state_parameters_shape
    
        samples = []

        for (idx, x) in enumerate(X)

            if isnothing(features)
                n_time_steps = size(x, 2)
            else
                n_time_steps = length(x)
            end

            push!( samples, Sample(
                x, y[idx],
                Array{Float64}(undef, model.state_parameters_shape),
                Vector{Float64}(undef, length(model.transitions) ),
                Array{Float64}(undef, n_time_steps + 1, n_states, n_classes),
                Array{Float64}(undef, n_time_steps, n_states, n_states, n_classes),
                Array{Float64}(undef, n_time_steps + 1, n_states, n_classes),
                0
            ) )

        end
    
        new{typeof(features)}(model, features, samples,
                    Vector{Task}(undef, length(samples)), one_task_per_sample)
    
    end

end
