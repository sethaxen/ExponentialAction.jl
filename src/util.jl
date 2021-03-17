# workaround for opnorm(::AbstractVector, Inf) not being implemented
_opnormInf(B) = opnorm(B, Inf)
_opnormInf(B::AbstractVector) = norm(B, Inf)

# we only use _opnormInf for control flow, so avoid differentiating through it
ChainRulesCore.@non_differentiable _opnormInf(B)

# TODO: replace with estimate
# see https://github.com/JuliaLang/julia/pull/39058
opnormest1(A) = opnorm(A, 1)
