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
    t = t_old = ts[begin]
    F = expv(t, A, B; shift, tol) * exp(μ * t)
    Fs = [F]
    for t in ts[(begin + 1):end]
        Δt = t - t_old
        F = expv(Δt, A, F; shift, tol) * exp(μ * Δt)
        Fs = vcat(Fs, [F])  # avoid mutation
        t_old = t
    end
    return Fs
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

    if num_steps == 0
        return [F1]
    elseif num_steps ≤ scale
        return vcat([F1], _expv_sequence_core1(Δt, A, F1, degree_opt, μ, num_steps, tol))
    else
        num_steps_per_scale = fld(num_steps, scale)  # d
        scale_actual, num_steps_last = fldmod(num_steps, num_steps_per_scale)  # (j, r)
        Fs = [F1]
        for i in 1:(scale_actual + 1)
            d = i > scale_actual ? num_steps_last : num_steps_per_scale
            d == 0 && continue
            Fs = vcat(Fs, _expv_sequence_core2(Δt, A, Fs[end], degree_opt, μ, d, tol))
        end
        return Fs
    end
end

function _expv_sequence_core1(Δt, A, B, degree_opt, μ, num_steps, tol)
    η = exp(Δt * μ)
    F = expv_taylor(Δt, A, B, degree_opt; tol) * η
    Fs = [F]
    for _ in 2:num_steps
        F = expv_taylor(Δt, A, F, degree_opt; tol) * η
        Fs = vcat(Fs, [F])  # avoid mutation
    end
    return Fs
end

function _expv_sequence_core2(Δt, A, B, degree_opt, μ, num_steps, tol)
    F1, Zs = expv_taylor_cache(Δt, A, B, degree_opt, 1, [B]; tol)
    F1 *= exp(μ * Δt)
    Fs = [F1]
    for k in 2:num_steps
        F, Zs = expv_taylor_cache(Δt, A, B, degree_opt, k, Zs; tol)
        F *= exp(μ * k * Δt)
        Fs = vcat(Fs, [F])  # avoid mutation
    end
    return Fs
end
