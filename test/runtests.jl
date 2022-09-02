using ExponentialAction
using Random
using Test

Random.seed!(1)  # set seed used for all testsets

@testset "ExponentialAction.jl" begin
    include("util.jl")
    include("taylor.jl")
    include("expv.jl")
    include("sequence.jl")
    include("autodiff.jl")
end