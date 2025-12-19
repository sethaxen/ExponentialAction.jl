"""
`ExponentialAction.jl` implements the action of the [Matrix exponential](https://en.wikipedia.org/wiki/Matrix_exponential).

For details, see the docstring of [`expv`](@ref).

# Exports

- [`expv`](@ref): compute the action of the matrix exponential
"""
module ExponentialAction

const MINIMUM_TOLERANCE = exp2(-precision(Float64))

using LinearAlgebra
using ChainRulesCore: ChainRulesCore
using SparseArrays: sparse
using AbstractDifferentiation: AbstractDifferentiation as AD

include("util.jl")
include("coefficients.jl")
include("parameters.jl")
include("taylor.jl")
include("expv.jl")
include("sequence.jl")

export expv, expv_sequence

end
