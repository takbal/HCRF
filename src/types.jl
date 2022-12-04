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

struct ObjectiveFunc{T <: AbstractArray}
    model::HCRFModel
    X::Vector{Vector{Int64}}
    y::Vector{Int64}
    features::T
    state_gradients::Vector{Array{Float64,3}}
    transition_gradients::Vector{Vector{Float64}}
    forward_tables::Vector{Array{Float64,3}}
    transition_tables::Vector{Array{Float64,4}}
    backward_tables::Vector{Array{Float64,3}}

    function ObjectiveFunc(model, X, y, features)

        gradients = []
        state_gradients = []
        transition_gradients = []
    
        forward_tables = []
        transition_tables = []
        backward_tables = Array{Float64,3}[]

        n_states, n_classes, _ = model.state_parameters_shape
        
        # pre-allocate target gradient and transition tables for each sample

        for i in eachindex(X)

            if isnothing(features)
                n_time_steps = size(X[i], 2)
            else
                n_time_steps = length(X[i])
            end

            state_gradient = Array{Float64}(undef, model.state_parameters_shape)
            transition_gradient = Vector{Float64}(undef, length(model.transitions) )
    
            push!(state_gradients, state_gradient)
            push!(transition_gradients, transition_gradient)

            # Add extra 1 time step for start state
            push!(forward_tables, Array{Float64}(undef, n_time_steps + 1, n_states, n_classes))
            push!(transition_tables, Array{Float64}(undef, n_time_steps, n_states, n_states, n_classes))
            push!(backward_tables, Array{Float64}(undef, n_time_steps + 1, n_states, n_classes))
        end
    
        new{typeof(features)}(model, X, y, features, state_gradients, transition_gradients,
                   forward_tables, transition_tables, backward_tables)
    
    end

end
