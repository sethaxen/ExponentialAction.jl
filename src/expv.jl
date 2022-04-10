"""
    expv(t, A, B; shift=true, tol)

Compute ``\\exp(tA)B`` without computing ``tA`` or the matrix exponential  ``\\exp(tA)``.

Computing the action of the matrix exponential is significantly faster than computing
the matrix exponential and then multiplying it when the second dimension of ``B`` is
much smaller than the first one. The "time" ``t`` may be real or complex.

In short, the approach computes
```math
F = T_m(tA / s)^s B,
```
where ``T_m(X)`` is the Taylor series of ``\\exp(X)`` truncated to degree ``m = m^*``.
The term ``s`` determines how many times the Taylor series acts on ``B``.
``m^*`` and ``s`` are chosen to minimize the number of matrix products needed while
maintaining the required tolerance `tol`.

The algorithm is described in detail in Algorithm 3.2 in [^AlMohyHigham2011].

[^AlMohyHigham2011]: Al-Mohy, Awad H. and Higham, Nicholas J. (2011)
    Computing the Action of the Matrix
    Exponential, with an Application to Exponential Integrators. SIAM Journal on Scientific
    Computing, 33 (2). pp. 488-511. ISSN 1064-8275
    doi: [10.1137/100788860](https://doi.org/10.1137/100788860)
    eprint: [eprints.maths.manchester.ac.uk/id/eprint/1591](http://eprints.maths.manchester.ac.uk/id/eprint/1591)

# Keywords

  - `shift=true`: Expand the Taylor series about the ``n \\times n`` matrix ``A-μI=0``
    instead of ``A=0``, where ``μ = \\operatorname{tr}(A) / n`` to speed up convergence. See
    §3.1 of [^AlMohyHigham2011].
  - `tol`: The relative tolerance at which to compute the result. Defaults to the tolerance
    of the eltype of the result.
"""
function expv(t, A, B; shift=true, tol=default_tol(t, A, B))
    A, μ = shift ? shift_matrix(A) : (A, zero(float(eltype(A))))
    degree_opt, scale = parameters(t, A, size(B, 2); tol)  # m*, s
    F = _expv_core(t * one(μ) / scale, A, B, degree_opt, μ, scale, tol)
    return F
end
expv(t, A::Diagonal, B; kwargs...) = exp.(t .* A.diag) .* B

function _expv_core(Δt, A, B, degree_opt, μ, num_steps, tol)
    η = exp(Δt * μ)
    F = one(η) * B
    for _ in 1:num_steps
        F = expv_taylor(Δt, A, F, degree_opt; tol) * η  # new starting matrix
    end
    return F
end
