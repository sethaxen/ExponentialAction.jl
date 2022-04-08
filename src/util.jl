# workaround for opnorm(::AbstractVector, Inf) not being implemented
_opnormInf(B) = opnorm(B, Inf)
_opnormInf(B::AbstractVector) = norm(B, Inf)

# TODO: replace with estimate
# see https://github.com/JuliaLang/julia/pull/39058
opnormest1(A) = opnorm(A, 1)

asint(x) = x ≤ typemax(Int) ? Int(x) : typemax(Int)

default_tolerance(args...) = eps(float(real(Base.promote_eltype(args...))))

# we only use these for control flow, so avoid differentiating through them
ChainRulesCore.@non_differentiable _opnormInf(B)
ChainRulesCore.@non_differentiable default_tolerance(args...)
