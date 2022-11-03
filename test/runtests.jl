using Test, HCRF, Optim, Random, SparseArrays

Random.seed!(123456)

#---------------------------------------------------

@testset "test_train_regression" begin

    X = [ [ 1.0 2 ; 5 9 ; 7 3 ],
          [ 6 -2 ; 3 3.0 ],
          [ 1 -1.0 ],
          [ 1 1 ; 5 3 ; 4 2 ; 3.0 3 ] ]

    y = [ 0, 1, 0, 1 ]

    model, result = HCRF.fit!(X, y, num_states = 3)
    actual = predict(model, X)

    @test isequal( predict(model, X), y)
end

#---------------------------------------------------

@testset "test_train_regression_sparse" begin
    X = [
        sparse(1:3, [2; 5; 8], ones(3), 3, 10),
        sparse(1:3, [4; 9; 8], ones(3), 3, 10),
        sparse([1], [4], ones(1), 1, 10),
        sparse(1:4, [2; 5; 8; 10], ones(4), 4, 10),
    ]

    y = [ 1, 1, 0, 1 ]

    model, result = HCRF.fit!(X, y, num_states = 5, L1_penalty = 0., L2_penalty = 1. )
    actual = predict(model, X)

    @test isequal( predict(model, X), y)

end

#---------------------------------------------------

@testset "test_forward_backward" begin

    transitions = [
            [1, 1, 2],
            [2, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
            [1, 2, 1],
            [2, 2, 1],
        ]
    transition_parameters = [1., 0, 2, 1, 3, -2]
    x = [ 2. 3 ; 1. 0 ; 0. 2]
    state_parameters = [ -1. 1 ; 0. 2 ;;; 1. -1 ; -2. 3 ]

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

    x_dot_parameters = reshape(x * reshape(state_parameters, n_features, :), n_time_steps, n_states, n_classes)

    forward_table, forward_transition_table, backward_table = HCRF.forward_backward(
        x_dot_parameters, state_parameters, transition_parameters, transitions
    )

    @test forward_table ≈ expected_forward_table atol=0.0001
    @test forward_table[end,end,1] ≈ backward_table[1,1,1] atol=0.0001
    @test forward_table[end,end,2] ≈ backward_table[1,1,2] atol=0.0001

end

#---------------------------------------------------

@testset "test_gradient_small_transition" begin

    transitions = [ [1, 1, 1], [2, 1, 1] ]
    transition_parameters =[1. ; 0]
    x = [2. 3 -1]
    state_parameters = [ -1. ; 5 ; 2 ;;; 2 ; -6 ; 13 ]
    cy = 2

    delta = 5.0 ^ -5

    dstate_gradient = similar(state_parameters)
    dtransition_gradient = similar(transition_parameters)

    for trans in eachindex(transition_parameters)

        tpd = fill!(similar(transition_parameters), 0.0)
        tpd[trans] = delta
        ll0 = HCRF.log_likelihood(x, cy, state_parameters, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        dtp0 = dtransition_gradient
        ll1 = HCRF.log_likelihood(x, cy, state_parameters, transition_parameters + tpd, transitions, dstate_gradient, dtransition_gradient)
        expected_der = (ll1 - ll0) / delta
        actual_der = dtp0[trans]

        @test expected_der ≈ actual_der atol=0.01
    end

end

#---------------------------------------------------

@testset "test_gradient_small" begin

    transitions = [
        [1, 1, 2],
        [2, 1, 2],
        [1, 2, 2],
        [2, 2, 2],
        [1, 2, 1],
        [2, 2, 1],
    ]

    transition_parameters = [1., 0, 2, 1, 3, -2]
    x = [2. 3 -1]
    state_parameters = [ -1. 3 ; 5 7 ; -3 2 ;;; 2. -4 ; -6 8 ; 6 13 ]
    cy = 2

    delta = 5.0 ^ -5

    dstate_gradient = similar(state_parameters)
    dtransition_gradient = similar(transition_parameters)

    for trans in eachindex(transition_parameters)

        tpd = fill!(similar(transition_parameters), 0.0)
        tpd[trans] = delta
        ll0 = HCRF.log_likelihood(x, cy, state_parameters, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        dtp0 = copy(dtransition_gradient)
        ll1 = HCRF.log_likelihood(x, cy, state_parameters, transition_parameters + tpd, transitions, dstate_gradient, dtransition_gradient)
        expected_der = (ll1 - ll0) / delta
        actual_der = dtp0[trans]

        @test expected_der ≈ actual_der atol=0.01
    end

end

#---------------------------------------------------

@testset "test_gradient_large_state" begin

    transitions = [
        [1, 1, 2],
        [2, 1, 2],
        [1, 2, 2],
        [2, 2, 2],
        [1, 2, 1],
        [2, 2, 1],
    ]

    transition_parameters = [1., 0, 2, 1, 3, -2]
    x = [ 2. 3 -1 ; 1. 4 -2 ; 5. 2 -3 ;  -2. 5  3 ]
    state_parameters = [ -1. 3 ; 5 7 ; -3 2 ;;; 2. -4 ; -6 8 ; 6 13 ]
    cy = 2
    delta = 5.0 ^ -4

    dstate_gradient = similar(state_parameters)
    dtransition_gradient = similar(transition_parameters)

    for idx in CartesianIndices(state_parameters)

        spd = fill!(similar(state_parameters), 0.0)
        spd[idx] = delta
        ll0 = HCRF.log_likelihood(x, cy, state_parameters, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        dsp0 = copy(dstate_gradient)
        ll1 = HCRF.log_likelihood(x, cy, state_parameters + spd, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        expected_der = (ll1 - ll0) / delta
        actual_der = dsp0[idx]

        @test expected_der ≈ actual_der atol=0.01
    end
end

#---------------------------------------------------

@testset "test_gradient_large_transition" begin

    transitions = [
        [1, 1, 2],
        [2, 1, 2],
        [1, 2, 2],
        [2, 2, 2],
        [1, 2, 1],
        [2, 2, 1],
    ]

    transition_parameters = [1. -5 20 1 3 -2]
    x = [ 2. 3 -1 ; 1. 4 -2 ; 4. -4 2 ;  3 5  3 ]
    state_parameters = [ -1. 3 ; 5 7 ; -3 2 ;;; 2. -4 ; -6 8 ; 6 13 ]
    cy = 2
    delta = 5.0 ^ -5

    dstate_gradient = similar(state_parameters)
    dtransition_gradient = similar(transition_parameters)

    for trans in eachindex(transition_parameters)

        tpd = fill!(similar(transition_parameters), 0.0)
        tpd[trans] = delta
        ll0 = HCRF.log_likelihood(x, cy, state_parameters, transition_parameters, transitions, dstate_gradient, dtransition_gradient)
        dtp0 = copy(dtransition_gradient)
        ll1 = HCRF.log_likelihood(x, cy, state_parameters, transition_parameters + tpd, transitions, dstate_gradient, dtransition_gradient)
        expected_der = (ll1 - ll0) / delta
        actual_der = dtp0[trans]

        @test expected_der ≈ actual_der atol=0.01
    end

end
