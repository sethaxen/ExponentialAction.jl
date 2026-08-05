using ExponentialAction
using Random
using Test

Random.seed!(1)  # set seed used for all testsets

@testset "ExponentialAction.jl" begin
    include("aqua.jl")
    include("helpers.jl")
    include("util.jl")
    include("opnorm_est.jl")
    include("trace_est.jl")
    include("taylor.jl")
    include("expv.jl")
    include("sequence.jl")
    include("autodiff.jl")
    include("gpu.jl")
end
