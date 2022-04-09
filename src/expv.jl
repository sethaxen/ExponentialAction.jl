"""
    expv(t, A, B; kwargs...)

Compute `exp(t*A) * B` without computing `t * A` or the matrix exponential.
This is significantly faster than the matrix exponential when the second dimension of `B` is
much smaller than the first one. The "time" `t` may be real or complex.
The algorithm is described in [^AlMohyHigham2011].

# Keywords

  - `shift=true`: Expand the Taylor series of `exp(t*A)` about ``A-μI=0`` instead of
    ``A=0``, where ``μ = \\operatorname{tr}(A) / n`` to speed up convergence. See
    §3.1 of [^AlMohyHigham2011].
  - `tol`: The tolerance at which to compute the result. Defaults to the tolerance of the
    eltype of the result.

[^AlMohyHigham2011]: Al-Mohy, Awad H. and Higham, Nicholas J. (2011) Computing the Action of the Matrix
    Exponential, with an Application to Exponential Integrators. SIAM Journal on Scientific
    Computing, 33 (2). pp. 488-511. ISSN 1064-8275
    doi: [10.1137/100788860](https://doi.org/10.1137/100788860)
    eprint: [eprints.maths.manchester.ac.uk/id/eprint/1591](http://eprints.maths.manchester.ac.uk/id/eprint/1591)
"""
function expv(t, A, B; shift=true, tol=default_tol(t, A, B))
    n = LinearAlgebra.checksquare(A)
    # §3: “Our experience indicates that p_max = 8 and m_max = 55 are appropriate choices.”
    p_max = 8
    m_max = 55
    n0 = size(B, 2)
    if shift
        μ = tr(A) / n
        A -= μ * I
    else
        μ = zero(float(eltype(A)))
    end
    params = parameters(t, A, n0, m_max, p_max, tol)
    η = exp(t * μ / params.s)
    F = one(η) * B
    for i in 1:(params.s)
        c1 = _opnormInf(B)
        for j in 1:(params.m)
            B = (A * B) * (t / (params.s * j))
            c2 = _opnormInf(B)
            F += B
            c1 + c2 ≤ tol * _opnormInf(F) && break
            c1 = c2
        end
        F *= η
        B = F
    end
    return F
end
expv(t, A::Diagonal, B; kwargs...) = exp.(t .* A.diag) .* B
