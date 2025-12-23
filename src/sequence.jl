"""
    expv_sequence(t::AbstractVector, A, B; kwargs...)

Compute ``\\exp(t_i A)B`` for the (sorted) sequence of (real) time points ``t=(t_1, t_2, \\ldots)``.

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

    expv_sequence(t::AbstractRange, A, B; kwargs...)

Compute [`expv`](@ref) over the uniformly spaced sequence.

This algorithm takes special care to avoid overscaling and to save and reuse matrix products
and is described in Algorithm 5.2 of [^AlMohyHigham2011].
"""
function expv_sequence(ts, A, B; shift = true, tol = default_tol(ts, A, B))
    A, μ = shift ? shift_matrix(A) : (A, zero(float(eltype(A))))
    t = t_old = ts[begin]
    F = expv(t, A, B; shift, tol) * exp(μ * t)
    Fs = [F]
    for t in ts[(begin + 1):end]
        Δt = t - t_old
        F = expv(Δt, A, F; shift = false, tol) * exp(μ * Δt)
        Fs = vcat(Fs, [F])  # avoid mutation
        t_old = t
    end
    return Fs
end
function expv_sequence(ts::AbstractRange, A, B; shift = true, tol = default_tol(ts, A, B))
    ncols_B = size(B, 2)
    num_steps = length(ts) - 1  # q
    t_min = ts[begin]
    t_max = ts[end]
    t_span = t_max - t_min
    Δt = (t_max - t_min) / num_steps

    A, μ = shift ? shift_matrix(A) : (A, zero(float(eltype(A))))

    degree_opt, scale = parameters(t_span, A, ncols_B; tol)  # (m*, s)

    F = expv(t_min, A, B; shift = false, tol) * exp(μ * t_min)

    if num_steps == 0
        return [F]
    elseif num_steps ≤ scale
        # points are intermediate matrices of a scaling sequence
        _, Fs_tail = _expv_sequence_core(Δt, A, F, degree_opt, μ, num_steps, tol)
        return vcat([F], Fs_tail)
    else
        num_steps_per_scale = fld(num_steps, scale)  # d
        scale_actual, num_steps_last = fldmod(num_steps, num_steps_per_scale)  # (j, r)
        Fs = [F]
        for i in 1:(scale_actual + 1)
            # for each scaling step, points are intermediate matrices
            # get number of steps to take in this scale
            d = i ≤ scale_actual ? num_steps_per_scale : num_steps_last
            d == 0 && continue
            F, Fs_tail = _expv_sequence_core_cache(Δt, A, F, degree_opt, μ, d, tol)
            Fs = vcat(Fs, Fs_tail)
        end
        return Fs
    end
end

# like _expv_core, but returns the entire sequence of F matrices
function _expv_sequence_core(Δt, A, B, degree_opt, μ, num_steps, tol)
    η = exp(Δt * μ)
    F = expv_taylor(Δt, A, B, degree_opt; tol) * η
    Fs = [F]
    for _ in 2:num_steps
        F = expv_taylor(Δt, A, F, degree_opt; tol) * η  # new starting matrix
        Fs = vcat(Fs, [F])  # avoid mutation
    end
    return F, Fs
end

# like _expv_sequence_core, but all steps are relative to B, so we can cache matrix products
function _expv_sequence_core_cache(Δt, A, B, degree_opt, μ, num_steps, tol)
    F1, Zs = expv_taylor_cache(Δt, A, B, degree_opt, 1, [B]; tol)
    F = F1 *= exp(μ * Δt)
    Fs = [F1]
    for k in 2:num_steps
        F, Zs = expv_taylor_cache(Δt, A, B, degree_opt, k, Zs; tol)
        F *= exp(μ * k * Δt)
        Fs = vcat(Fs, [F])  # avoid mutation
    end
    return F, Fs
end
