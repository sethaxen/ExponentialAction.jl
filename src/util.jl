# workaround for opnorm(::AbstractVector, Inf) not being implemented
_opnormInf(B) = opnorm(AD.primal_value(B), Inf)
_opnormInf(B::AbstractVector) = norm(AD.primal_value(B), Inf)

# TODO: replace with estimate
# see https://github.com/JuliaLang/julia/pull/39058
opnormest1(A) = opnorm(A, 1)

asint(x) = x ≤ typemax(Int) ? Int(x) : typemax(Int)

default_tol(args...) = AD.primal_value(eps(float(real(Base.promote_eltype(args...)))))

function shift_matrix(A)
    μ = get_shift(A)
    return (A - μ * I), μ
end

function _col_normalize!(x)
    foreach(LinearAlgebra.normalize!, eachcol(x))
    return x
end

function _rand_sphere!(rng, x)
    Random.randn!(rng, x)
    _col_normalize!(x)
    return x
end

function _rand_signs!(rng::Random.AbstractRNG, x::AbstractArray{<:Real})
    Random.rand!(rng, x)
    x .= 2 .* (x .> 0.5) .- 1
    return x
end
function _rand_signs!(rng::Random.AbstractRNG, x::AbstractArray{T}) where {T <: Complex}
    Random.randn!(rng, x)
    x .= ifelse.(iszero.(x), oneunit(T), sign.(x))
    return x
end

get_shift(A) = AD.primal_value(tr(A)) / LinearAlgebra.checksquare(A)
_similar_to_prototype(::Nothing, T::Type, axes) = similar(Array{T}, axes)
_similar_to_prototype(prototype::AbstractArray, T::Type, axes) = similar(prototype, T, axes)

# builds axes for the test matrix or a cache. Always prefer axes from the prototype if possible.
_build_axes(operator, prototype::AbstractVecOrMat) = axes(prototype, 1)
_build_axes(operator::AbstractMatrix, ::Nothing) = axes(operator, 2)
_build_axes(operator, ::Nothing) = Base.OneTo(size(operator, 2))
_build_axes(operator, prototype, length) = Base.OneTo(length)

# we only use these for control flow, so avoid differentiating through them
ChainRulesCore.@non_differentiable _opnormInf(B)
ChainRulesCore.@non_differentiable default_tol(args...)
ChainRulesCore.@non_differentiable get_shift(A)
