using HCRF, BenchmarkTools, JLD2

data = load(ARGS[1])

display(@benchmark HCRF.fit(data["X"], data["y"]; features = data["features"], num_states=6, suppress_warning = true, iterations = 10))
