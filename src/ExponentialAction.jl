module ExponentialAction

using LinearAlgebra
using ChainRulesCore: ChainRulesCore
using SparseArrays: sparse
using Compat: findmin
using AbstractDifferentiation: AD

include("util.jl")
include("coefficients.jl")
include("parameters.jl")
include("expv.jl")

export expv

end
