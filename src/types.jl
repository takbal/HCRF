
mutable struct HCRFModel
    # parameters used in the fit! that produced this model
    num_states::UInt
    L1_penalty::Float64 # L1 penalty constant used in the fit
    L2_penalty::Float64 # L2 penalty constant used in the fit
    use_L1_clipping::Bool # if false, no clipping was used in L1
    state_parameter_noise::Float64 # variance of init normal noise
    transition_parameter_noise::Float64 # variance of init normal noise
    transitions::Array{Vector{Int}}

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

mutable struct ObjectiveFunc{XT <: AbstractVector, yT <: AbstractVector}
    model::HCRFModel
    X::XT
    y::yT
end
