using Test, HCRF, Optim, Random, SparseArrays

Random.seed!(123456)

function gen_data()

    X = [ [ 1. 5. 7.
            2. 9. 3. ],
          [  6. 3.
            -2. 3. ],
          [ 1.
           -1. ],
          [ 1. 5. 4. 3.
            1. 3. 2. 3. ] ]

    y = [ 0, 1, 0, 1 ]

    return X, y
end

function gen_transitions()
    return [
            [1, 1, 2],
            [2, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
            [1, 2, 1],
            [2, 2, 1],
        ]
end

function get_x_dot_parameters(x, state_parameters)
    n_features = size(state_parameters,3)
    # return reshape(reshape(state_parameters, :, n_features) * x, n_states, n_classes, size(x,2))
    return reshape(state_parameters, :, n_features) * x
end

function forward_backward(x_dot_parameters, transition_parameters, transitions, n_states, n_classes)

    n_time_steps = size(x_dot_parameters,2)

    forward_table = Array{Float64}(undef, n_time_steps + 1, n_states, n_classes)
    transition_table = Array{Float64}(undef, n_time_steps, n_states, n_states, n_classes)
    backward_table = Array{Float64}(undef, n_time_steps + 1, n_states, n_classes)

    HCRF.forward!(forward_table, transition_table, x_dot_parameters, 1:size(x_dot_parameters,2), transition_parameters, transitions; need_transition_table = true)
    HCRF.backward!(backward_table, x_dot_parameters, 1:size(x_dot_parameters,2), transition_parameters, transitions)

    return forward_table, transition_table, backward_table

end

function ll(x, y, state_parameters, transition_parameters, transitions, state_gradient, transition_gradient)

    n_states, n_classes, _ = size(state_parameters)

    forward_table, transition_table, backward_table = forward_backward(
        get_x_dot_parameters(x, state_parameters), transition_parameters, transitions, n_states, n_classes)

    return HCRF.log_likelihood!(state_gradient, transition_gradient, x, 1:size(x,2), y, transitions,
        forward_table, transition_table, backward_table)

end

#---------------------------------------------------

@testset "test_train" begin

    X, y = gen_data()

    model, result = HCRF.fit!(X, y, num_states = 3; L1_penalty = 0.15, transition_generator = step_transitions)

    @test isequal( predict(model, X), y)
end

@testset "test_train_unconstrained" begin

    # test alternative feature generation
    X, y = gen_data()

    features = hcat(X...)
    X = [ [1;2;3], [4;5], [6], [7;8;9;10] ]

    model, result = HCRF.fit!(X, y; features, num_states = 3)

    @test isequal( predict(model, X; features), y)

    _, hidden = predict_marginals(model, X; features, calc_hidden = true)

    for h in hidden
        @test h[1] == 1 && h[end] == 3 && all(h[2:end-1] .== 2)
    end
end

#---------------------------------------------------

@testset "test_train_sparse" begin
    X = [
        sparse([2; 5; 8], 1:3, ones(3), 10, 3),
        sparse([4; 9; 8], 1:3, ones(3), 10, 3),
        sparse([4], [1], ones(1), 10, 1),
        sparse([2; 5; 8; 10], 1:4, ones(4), 10, 4),
    ]

    y = [ 1, 1, 0, 1 ]

    model, result = HCRF.fit!(X, y, num_states = 5, L1_penalty = 0., L2_penalty = 1. )
    actual = predict(model, X)

    @test isequal( predict(model, X), y)

end

#---------------------------------------------------

@testset "test_forward_backward" begin

    transitions = gen_transitions()

    transition_parameters = [1., 0, 2, 1, 3, -2]
    x = [ 2. 1  0 ; 3  0  2 ]
    state_parameters = [ -1. 1 ; 1. -1 ;;; 0. -2 ; 2. 3 ]

    A = zeros(4, 2, 2)

    A[1, 1, 1] = 1
    A[1, 1, 2] = 1
    A[1, 2, 1] = 0
    A[1, 2, 2] = 0

    A[2, 1, 1] = 0
    A[2, 1, 2] = 0
    A[2, 2, 1] = 1 * exp(1) * exp(8)
    A[2, 2, 2] = 1 * exp(0) * exp(7)

    A[3, 1, 1] = 1 * exp(1 + 8) * exp(3) * exp(-1)
    A[3, 1, 2] = 1 * exp(0 + 7) * exp(-2) * exp(1)
    A[3, 2, 1] = 1 * exp(1 + 8) * exp(2) * exp(1)
    A[3, 2, 2] = 1 * exp(0 + 7) * exp(1) * exp(-1)

    A[4, 1, 1] = 1 * exp(1 + 8) * exp(2 + 1) * exp(3) * exp(0)
    A[4, 1, 2] = 1 * exp(0 + 7) * exp(1 - 1) * exp(-2) * exp(-4)
    A[4, 2, 1] = 1 * exp(1 + 8 + 3 - 1) * exp(1 + 4) + 1 * exp(1 + 8 + 2 + 1) * exp(2 + 4)
    A[4, 2, 2] = 1 * exp(0 + 7 - 2 + 1) * exp(0 + 6) + 1 * exp(0 + 7 + 1 - 1) * exp(1 + 6)

    expected_forward_table = log.(A)
    
    n_features = 2
    n_time_steps = 3
    n_states = 2
    n_classes = 2

    forward_table, _, backward_table = forward_backward(
        get_x_dot_parameters(x, state_parameters), transition_parameters, transitions, n_states, n_classes)

    @test forward_table ≈ expected_forward_table atol=0.0001
    @test forward_table[end,end,1] ≈ backward_table[1,1,1] atol=0.0001
    @test forward_table[end,end,2] ≈ backward_table[1,1,2] atol=0.0001

end

#---------------------------------------------------

@testset "test_gradient_small_transition" begin

    transitions = [ [1, 1, 1], [2, 1, 1] ]
    transition_parameters =[1. ; 0]
    x = [2. ; 3 ; -1]
    state_parameters = [ -1. 2 ;;; 5 -6 ;;; 2 13 ]
    cy = 2

    delta = 5.0 ^ -5

    dstate_gradient = similar(state_parameters)
    dtransition_gradient = similar(transition_parameters)

    for trans in eachindex(transition_parameters)

        tpd = fill!(similar(transition_parameters), 0.0)
        tpd[trans] = delta
        ll0 = ll(x, cy, state_parameters, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        dtp0 = dtransition_gradient
        ll1 = ll(x, cy, state_parameters, transition_parameters + tpd, transitions, dstate_gradient, dtransition_gradient)
        expected_der = (ll1 - ll0) / delta
        actual_der = dtp0[trans]

        @test expected_der ≈ actual_der atol=0.01
    end

end

#---------------------------------------------------

@testset "test_gradient_small" begin

    transitions = gen_transitions()

    transition_parameters = [1., 0, 2, 1, 3, -2]
    x = [2. ; 3 ; -1]
    state_parameters = [ -1. 2 ; 3 -4 ;;; 5 -6 ; 7 8 ;;; -3 6 ; 2 13 ]
    cy = 2

    delta = 5.0 ^ -5

    x_dot_parameters = get_x_dot_parameters(x, state_parameters)

    dstate_gradient = similar(state_parameters)
    dtransition_gradient = similar(transition_parameters)

    for trans in eachindex(transition_parameters)

        tpd = fill!(similar(transition_parameters), 0.0)
        tpd[trans] = delta
        ll0 = ll(x, cy, state_parameters, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        dtp0 = copy(dtransition_gradient)
        ll1 = ll(x, cy, state_parameters, transition_parameters + tpd, transitions, dstate_gradient, dtransition_gradient)
        expected_der = (ll1 - ll0) / delta
        actual_der = dtp0[trans]

        @test expected_der ≈ actual_der atol=0.01
    end

end

#---------------------------------------------------

@testset "test_gradient_large_state" begin

    transitions = gen_transitions()

    transition_parameters = [1., 0, 2, 1, 3, -2]
    x = [ 2. 1 5 -2 ; 3 4 2 5 ; -1 -2 -3 3 ]
    state_parameters = [ -1. 2 ; 3 -4 ;;; 5 -6 ; 7 8 ;;; -3 6 ; 2 13 ]
    cy = 2
    delta = 5.0 ^ -4

    dstate_gradient = similar(state_parameters)
    dtransition_gradient = similar(transition_parameters)

    for idx in CartesianIndices(state_parameters)

        spd = fill!(similar(state_parameters), 0.0)
        spd[idx] = delta
        ll0 = ll(x, cy, state_parameters, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        dsp0 = copy(dstate_gradient)
        ll1 = ll(x, cy, state_parameters + spd, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        expected_der = (ll1 - ll0) / delta
        actual_der = dsp0[idx]

        @test expected_der ≈ actual_der atol=0.01
    end
end

#---------------------------------------------------

@testset "test_gradient_large_transition" begin

    transitions = gen_transitions()

    transition_parameters = [1., -5, 20, 1, 3, -2]
    x = [ 2. 1 4 3 ; 3 4 -4 5 ; -1 -2 2 3 ]
    state_parameters = [ -1. 2 ; 3 -4 ;;; 5 -6 ; 7 8 ;;; -3 6 ; 2 13 ]
    cy = 2
    delta = 5.0 ^ -5

    x_dot_parameters = get_x_dot_parameters(x, state_parameters)

    dstate_gradient = similar(state_parameters)
    dtransition_gradient = similar(transition_parameters)

    for trans in eachindex(transition_parameters)

        tpd = fill!(similar(transition_parameters), 0.0)
        tpd[trans] = delta
        ll0 = ll(x, cy, state_parameters, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        dtp0 = copy(dtransition_gradient)
        ll1 = ll(x, cy, state_parameters, transition_parameters + tpd, transitions, dstate_gradient, dtransition_gradient)
        expected_der = (ll1 - ll0) / delta
        actual_der = dtp0[trans]

        @test expected_der ≈ actual_der atol=0.01
    end

end
