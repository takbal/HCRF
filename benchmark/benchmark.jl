using HCRF, BenchmarkTools, JLD2, Jtb

# needs env with UniqueVectors, AxisKeys
data = load("data.jld2")

# println("running warmup ...")

# HCRF.fit!(data["warmup_X"], data["warmup_y"]; observations = data["warmup_obs"], num_states=6, suppress_warning = true, iterations = 1)

# println("running benchmark ...")

display(@benchmark HCRF.fit!(data["X"], data["y"]; features = data["obs"], num_states=6, suppress_warning = true, iterations = 10))
