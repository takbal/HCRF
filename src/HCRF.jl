module HCRF

    using Optim, SparseArrays

    include("types.jl")

    include("util.jl")

    include("objfunc.jl")

    include("model.jl")

    export HCRFModel, fit, predict, predict_marginals,
        step_transitions, unconstrained_transitions

end # module
