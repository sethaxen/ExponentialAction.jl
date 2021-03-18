"""
    parameters(t, A, n0, m_max[, p_max[, tol]]) -> NamedTuple{(:m, :s)}

Compute Taylor series parameters needed for `exp(t*A) * B`.

This is Code Fragment 3.1 from [^AlMohyHigham2011].
"""
function parameters(
    t, A, n0, m_max, p_max=p_from_m(m_max), tol=eps(float(real(Base.promote_eltype(t, A))))
)
    θ = coefficients(tol)
    T = float(Base.promote_eltype(t, A, θ))
    Tr = real(T)
    tnorm = abs(t * one(Tr))
    iszero(tnorm) && return (m=0, s=1)
    Anorm = Tr(opnormest1(A))
    iszero(Anorm) && return (m=0, s=1)
    tAnorm = tnorm * Anorm
    ℓ = 2 # §3: “where the positive integer ℓ is a parameter (typically set to 1 or 2)”
    if tAnorm * (n0 * m_max) ≤ θ[m_max] * (2 * ℓ * p_max * (p_max + 3)) # (3.13) is satisfied
        m_opt = 0
        Cm_opt = typemax(Int)
        # work around argmin not taking a function
        for m in 1:m_max
            Cm = m * _cld(tAnorm, θ[m])
            if Cm < Cm_opt
                m_opt = m
                Cm_opt = Cm
            end
        end
        s = _cld(Cm_opt, m_opt)
    else
        # TODO: replace powers of A here and below with opnormest(pow, A, 1)
        # see https://github.com/JuliaLang/julia/pull/39058
        Apow = A * A
        d = tnorm * sqrt(Tr(opnormest1(A)))
        m_opt = m_max + 1
        Cm_opt = typemax(Int)
        for p in 2:p_max # Compute minimum in (3.11)
            Apow *= A
            # (3.7)
            d, d_old = tnorm * Tr(opnormest1(Apow))^(1//(p + 1)), d
            α = max(d, d_old)
            m_min = p * (p - 1) - 1
            # work around argmin not taking a function
            for m in m_min:m_max
                Cm = m * _cld(α, θ[m])
                if Cm < Cm_opt || (Cm == Cm_opt && m < m_opt)
                    m_opt = m
                    Cm_opt = Cm
                end
            end
        end
        s = max(_cld(Cm_opt, m_opt), 1)
    end
    return (m=m_opt, s=s)
end
# work around opnorm(A, 1) and (A^2)*A having very slow defaults for these arrays
# https://github.com/sethaxen/ExponentialAction.jl/issues/3
function parameters(
    t,
    A::Union{Bidiagonal,Tridiagonal},
    n0,
    m_max,
    p_max=p_from_m(m_max),
    tol=eps(float(real(Base.promote_eltype(t, A)))),
)
    return parameters(t, sparse(A), n0, m_max, p_max, tol)
end

# avoid differentiating through parameters with ChainRules-compatible ADs
ChainRulesCore.@non_differentiable parameters(t, A, n0, m_max, p_max, tol)

# solution to p(p-1) ≤ m + 1
p_from_m(m) = oftype(m, fld(1 + sqrt(5 + 4 * m), 2))
