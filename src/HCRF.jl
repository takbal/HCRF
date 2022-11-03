module HCRF

include("types.jl")

include("objfunc.jl")

include("model.jl")
export HCRFModel, fit!, predict, predict_marginals

end # module
