using Revise
using JuliaInterpreter
using BenchmarkTools
using Optim
using LoopVectorization
using Test
using CUDA
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
optimize(f, [0. 0.], LBFGS())

