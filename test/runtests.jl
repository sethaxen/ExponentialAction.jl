using ExponentialAction
using Random
using Test

Random.seed!(1)  # set seed used for all testsets

@testset "ExponentialAction.jl" begin
    include("util.jl")
    include("expv.jl")
    include("expv_autodiff.jl")
end
