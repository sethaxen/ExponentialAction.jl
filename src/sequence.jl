"""
    expv_sequence(ts::AbstractVector, A, B; kwargs...)

Compute ``\\exp(t_i A)B`` for the (sorted) sequence of (real) time points ``\\{t_i\\}``.

At each time point, the result ``F_i`` is computed as
```math
F_i = \\exp\\left((t_i - t_{i-1}) A\\right) F_{i - 1}
```
using [`expv`](@ref), where ``t_0 = 0`` and ``F_0 = B``.
For details, see Equation 5.2 of [^AlMohyHigham2011].

Because the cost of computing `expv` is related to the operator 1-norm of ``t_i A``, this
incremental computation is more efficient than computing `expv` separately for each time
point.

See [`expv`](@ref) for a description of acceptable `kwargs`.

    expv_sequence(ts::AbstractRange, A, B; kwargs...)

Compute `expv` over the uniformly spaced sequence.

This algorithm takes special care to avoid overscaling and to save and reuse matrix products
and is described in Algorithm 5.2 of [^AlMohyHigham2011].
"""
function expv_sequence(ts, A, B; shift=true, tol=default_tol(ts, A, B))
    A, μ = shift ? shift_matrix(A) : (A, zero(float(eltype(A))))
    Fs_ts = accumulate(ts; init=(B, zero(ts[begin]))) do (F, t_old), t
        Δt = t - t_old
        return expv(Δt, A, F; shift=false, tol) * exp(μ * Δt), t
    end
    return first.(Fs_ts)
end
function expv_sequence(ts::AbstractRange, A, B; shift=true, tol=default_tol(ts, A, B))
    ncols_B = size(B, 2)
    num_steps = length(ts) - 1  # q
    t_min = ts[begin]
    t_max = ts[end]
    t_span = t_max - t_min
    Δt = (t_max - t_min) / num_steps

    A, μ = shift ? shift_matrix(A) : (A, zero(float(eltype(A))))

    degree_opt, scale = parameters(t_span, A, ncols_B; tol)  # (m*, s)

    F1 = expv(t_min, A, B; shift=false, tol) * exp(μ * t_min)

    elseif num_steps ≤ scale
        return vcat([F1], _expv_sequence_core1(Δt, A, F1, degree_opt, μ, num_steps, tol))
    end

    num_steps_per_scale = fld(num_steps, scale)  # d
    scale_actual, num_steps_last = fldmod(num_steps, num_steps_per_scale)  # (j, r)

    Fs = reduce(1:(scale_actual + 1); init=[F1]) do Fs, i
        d = i > scale_actual ? num_steps_last : num_steps_per_scale
        return vcat(Fs, _expv_sequence_core2(Δt, A, Fs[end], degree_opt, μ, d, tol))
    end
    return Fs
end

function _expv_sequence_core1(Δt, A, B, degree_opt, μ, num_steps, tol)
    η = exp(Δt * μ)
    return accumulate(1:num_steps; init=B) do F, _
        return expv_taylor(Δt, A, F, degree_opt; tol) * η
    end
end

function _expv_sequence_core2(Δt, A, B, degree_opt, μ, num_steps, tol)
    init = B, [B]
    Fs_Zs = accumulate(1:num_steps; init) do (_, Zs), k
        F, Zs_new = expv_taylor_cache(Δt, A, B, degree_opt, k, Zs; tol)
        F *= exp(μ * k * Δt)
        return (F, Zs_new)
    end
    return first.(Fs_Zs)
end
