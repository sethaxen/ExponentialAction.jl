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

get_shift(A) = AD.primal_value(tr(A)) / LinearAlgebra.checksquare(A)

# we only use these for control flow, so avoid differentiating through them
ChainRulesCore.@non_differentiable _opnormInf(B)
ChainRulesCore.@non_differentiable default_tol(args...)
ChainRulesCore.@non_differentiable get_shift(A)
