"""
`ExponentialAction.jl` implements the action of the [Matrix exponential](https://en.wikipedia.org/wiki/Matrix_exponential).

For details, see the docstring of [`expv`](@ref).

# Exports

- [`expv`](@ref): compute the action of the matrix exponential
"""
module ExponentialAction

using ArrayInterface: ArrayInterface
using LinearAlgebra
using ChainRulesCore: ChainRulesCore
using Random: Random
using SparseArrays: sparse
using AbstractDifferentiation: AbstractDifferentiation as AD
using DocStringExtensions: TYPEDEF, TYPEDFIELDS

include("util.jl")
include("linear_operators.jl")
include("opnorm_est.jl")
include("coefficients.jl")
include("parameters.jl")
include("taylor.jl")
include("expv.jl")
include("sequence.jl")

export expv, expv_sequence

end
