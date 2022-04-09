"""
    expv(t, A, B; shift=true, tol)

Compute `exp(t*A) * B` without computing `t * A` or the matrix exponential.
This is significantly faster than the matrix exponential when the second dimension of `B` is
much smaller than the first one. The "time" `t` may be real or complex.

In short, the approach computes

```math
F = \\left(\\prod_{i=1}^s T_m(tA / s)\\right) B,
```

where ``T_m(X)`` is the Taylor series of `\\exp(X)` truncated to degree ``m = m^*``.
The term ``s`` determines how many times the Taylor series acts on ``B``.
``m^*`` and ``s`` are chosen to minimize the number of matrix products needed while
maintaining the required tolerance `tol`.

The algorithm is described in detail in Algorithm 3.2 in [^AlMohyHigham2011].

# Keywords

  - `shift=true`: Expand the Taylor series of `exp(t*A)` about ``A-μI=0`` instead of
    ``A=0``, where ``μ = \\operatorname{tr}(A) / n`` to speed up convergence. See
    §3.1 of [^AlMohyHigham2011].
  - `tol`: The relative tolerance at which to compute the result. Defaults to the tolerance
    of the eltype of the result.

[^AlMohyHigham2011]: Al-Mohy, Awad H. and Higham, Nicholas J. (2011)
    Computing the Action of the Matrix
    Exponential, with an Application to Exponential Integrators. SIAM Journal on Scientific
    Computing, 33 (2). pp. 488-511. ISSN 1064-8275
    doi: [10.1137/100788860](https://doi.org/10.1137/100788860)
    eprint: [eprints.maths.manchester.ac.uk/id/eprint/1591](http://eprints.maths.manchester.ac.uk/id/eprint/1591)
"""
function expv(t, A, B; shift=true, tol=default_tol(t, A, B))
    n = LinearAlgebra.checksquare(A)
    if shift
        μ = tr(A) / n
        A -= μ * I
    else
        μ = zero(float(eltype(A)))
    end
    degree_opt, scale = parameters(t, A, size(B, 2); tol=tol)  # m*, s
    τ = t * one(μ) / scale
    η = exp(τ * μ)  # term for undoing shifting
    F = one(η) * B
    for _ in 1:scale
        F = expv_taylor(τ, A, F, degree_opt; tol=tol)
        F *= η
    end
    return F
end
expv(t, A::Diagonal, B; kwargs...) = exp.(t .* A.diag) .* B

"""
    expv_taylor(t, A, B, degree_max; tol)

Compute `exp(t*A)*B` using the truncated Taylor series with degree ``m=`` `degree_max`.

Instead of computing the Taylor series ``T_m(tA)`` of the matrix exponential directly, its
action on `B` is computed instead.

The series is truncated early if

```math
\\frac{\\lVert \\exp(t A) B - T_m(tA) B \\rVert_1}{\\lVert T_m(tA) B \\rVert_1} \\le \\mathrm{tol},
```

where ``\\lVert X \\rVert_1`` is the operator 1-norm of the matrix ``X``.
This condition is only approximately checked.
"""
function expv_taylor(t, A, B, degree_max; tol=default_tol(t, A, B))
    F = Z = B
    norm_tail_old = _opnormInf(Z)
    for j in 1:degree_max
        Z = (A * Z) * (t / j)  # (t A)ʲ/j! * B
        norm_tail = _opnormInf(Z)
        F += Z
        # check if ratio of norm of tail and norm of series is below tolerance
        norm_tail = norm_tail_old + norm_tail
        norm_tail ≤ tol * _opnormInf(F) && break
        norm_tail_old = norm_tail
    end
    return F
end
