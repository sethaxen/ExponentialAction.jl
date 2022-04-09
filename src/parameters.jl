"""
    parameters(t, A, ncols_B; kwargs...) -> (degree_opt, scale)

Compute Taylor series parameters needed for ``\\exp(t*A) * B``.

This is Code Fragment 3.1 from [^AlMohyHigham2011].

# Keywords
- `tol`: the desired relative tolerance
- `degree_max=55`: the maximum degree of the truncated Taylor series that will be used. This
  is ``m_{\\mathrm{max}}`` in [^AlMohyHigham2011], where they recommend a value of 55 in §3.
- `ℓ=2`: the number of columns in the matrix that is multiplied for norm estimation (note:
  currently only used for control flow.). Recommended values are 1 or 2.

# Returns
- `degree_opt`: the degree of the truncated Taylor series that will be used. This is
  ``m^*`` in [^AlMohyHigham2011],
- `scale`: the amount of scaling ``s`` that will be applied to ``A``. The truncated Taylor
  series of ``\\exp(t A / s)`` will be applied ``s`` times to ``B``.
"""
function parameters(t, A, ncols_B; tol=default_tol(t, A), degree_max::Int=55, ℓ::Int=2)
    return _parameters(AD.primal_value(t), AD.primal_value(A), ncols_B, degree_max, ℓ, tol)
end

function _parameters(t, A, ncols_B, degree_max, ℓ, tol)
    t_norm = abs(t)
    iszero(t_norm) && return (0, 1)
    Anorm = opnormest1(A)
    iszero(Anorm) && return (0, 1)
    tA_norm = t_norm * Anorm
    T = float(real(Base.promote_eltype(t, A)))
    θ = coefficients(T(tol))
    p_max = p_from_degree_max(degree_max)
    if _cost_case1(tA_norm, ncols_B, degree_max, θ[degree_max]) ≤ _cost_case2(ℓ, p_max)  # (3.13) is satisfied
        degree_opt = 0
        num_mat_mul_opt, degree_opt = findmin(1:degree_max) do m
            return asint(m * cld(tA_norm, θ[m]))
        end
        scale = cld(num_mat_mul_opt, degree_opt)
    else
        # TODO: replace powers of A here and below with opnormest(pow, A, 1)
        # see https://github.com/JuliaLang/julia/pull/39058
        Aᵖ⁺¹ = A * A
        d = t_norm * sqrt(opnormest1(A))
        degree_opt = degree_max + 1
        num_mat_mul_opt = typemax(Int)
        for p in 2:p_max # Compute minimum in (3.11)
            Aᵖ⁺¹ *= A
            # (3.7)
            d, d_old = t_norm * opnormest1(Aᵖ⁺¹)^(1//(p + 1)), d
            α = max(d, d_old)
            degree_min = p * (p - 1) - 1
            for degree in degree_min:degree_max
                num_mat_mul = asint(degree * cld(α, θ[degree]))
                if num_mat_mul < num_mat_mul_opt ||
                    (num_mat_mul == num_mat_mul_opt && degree < degree_opt)
                    degree_opt = degree
                    num_mat_mul_opt = num_mat_mul
                end
            end
        end
        scale = max(cld(num_mat_mul_opt, degree_opt), 1)
    end
    return (degree_opt, scale)
end
# work around opnorm(A, 1) and (A^2)*A having very slow defaults for these arrays
# https://github.com/sethaxen/ExponentialAction.jl/issues/3
function _parameters(t, A::Union{Bidiagonal,Tridiagonal}, ncols_B, degree_max, ℓ, tol)
    return _parameters(t, sparse(A), ncols_B, degree_max, ℓ, tol)
end

# avoid differentiating through parameters with ChainRules-compatible ADs
ChainRulesCore.@non_differentiable parameters(t, A, ncols_B)

# solution p to p(p-1) ≤ m + 1
p_from_degree_max(degree_max) = Int(fld(1 + sqrt(5 + 4 * degree_max), 2))

# approximate number of matrix-vector products needed for case 1
_cost_case1(Anorm, n0, m, θm) = Anorm * (n0 * m) / θm

# number of matrix-vector products needed for case 2: estimating all opnorm1 calls
_cost_case2(ℓ, p_max) = 2 * ℓ * p_max * (p_max + 3)
