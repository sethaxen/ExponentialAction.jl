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
function expv(t, A, B; shift=true, tol=default_tolerance(t, A, B))
    n = LinearAlgebra.checksquare(A)
    if shift
        μ = tr(A) / n
        A -= μ * I
    else
        μ = zero(float(eltype(A)))
    end
    degree_opt, scale = parameters(t, A, size(B, 2); tol=tol)  # m*, s
    η = exp(t * μ / scale)  # term for undoing shifting
    F = one(η) * Z
    Z = B
    for i in 1:scale
        norm_tail_old = _opnormInf(Z)
        # compute F ← exp(t A / scale) * F
        # let Zᵢ = Z
        for j in 1:degree_opt  # use Taylor series of exp(t A / scale)
            Z = (A * Z) * (t / (scale * j))  # (t A)^j/j! * Zᵢ
            norm_tail = _opnormInf(Z)
            F += Z
            # check if ratio of norm of tail and norm of series is below tolerance
            norm_tail = norm_tail_old + norm_tail
            norm_tail ≤ tol * _opnormInf(F) && break
            norm_tail_old = norm_tail
        end
        F *= η
        Z = F
    end
    return F
end
expv(t, A::Diagonal, B; kwargs...) = exp.(t .* A.diag) .* B
