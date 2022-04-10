"""
    expv_taylor(t, A, B, degree_max; tol)

Compute ``\\exp(tA)B`` using the truncated Taylor series with degree ``m=`` `degree_max`.

Instead of computing the Taylor series ``T_m(tA)`` of the matrix exponential directly, its
action on ``B`` is computed instead.

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
        F += Z
        # check if ratio of norm of tail and norm of series is below tolerance
        norm_tail = _opnormInf(Z)
        norm_tail_tot = norm_tail_old + norm_tail
        norm_tail_tot ≤ tol * _opnormInf(F) && break
        norm_tail_old = norm_tail
    end
    return F
end

"""
    expv_taylor_cache(t, A, B, degree_max, k, Z; tol)

Compute ``\\exp(tkA)B`` using the truncated Taylor series with degree ``m=`` `degree_max`.

This method stores all matrix products in a cache `Z`, where
``Z_p = \\frac{1}{(p-1)!} (t A)^{p-1} B``.
This cache can be reused if ``k`` changes but ``t``, ``A``, and ``B`` are unchanged.

`Z` is a vector of arrays of the same shape as `B` and is not mutated; instead the
(possibly updated) cache is returned.

# Returns

  - `F::AbstractMatrix`: The action of the truncated Taylor series
  - `Z::AbstractVector`: The cache of matrix products of the same shape as `F`. If the cache
    is updated, then this is a different object than the input `Z`.

See [`expv_taylor`](@ref).
"""
function expv_taylor_cache(t, A, B, degree_max, k, Zs; tol=default_tol(t, A, B))
    F = Z = B
    norm_tail_old = _opnormInf(Z)
    kʲ = float(one(k))
    cache_size = length(Zs)
    for j in 1:degree_max
        kʲ *= k
        if cache_size < j + 1
            Z = (A * Z) * (t / j)  # (t A)ʲ/j! * B
            Zs = vcat(Zs, [Z])
        else
            Z = Zs[j + 1]
        end
        F = muladd(kʲ, Z, F)
        # check if ratio of norm of tail and norm of series is below tolerance
        norm_tail = kʲ * _opnormInf(Z)
        norm_tail_tot = norm_tail_old + norm_tail
        norm_tail_tot ≤ tol * _opnormInf(F) && break
        norm_tail_old = norm_tail
    end
    return F, Zs
end
