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
function expv_sequence(ts, A, B; kwargs...)
    F = B
    t_old = zero(ts[begin])
    Fs = map(ts) do t
        F = expv(t - t_old, A, F; kwargs...)
        t_old = t
        return F
    end
    return Fs
end
function expv_sequence(ts::AbstractRange, A, B; shift=true, tol=default_tol(ts, A, B))
    n = LinearAlgebra.checksquare(A)
    ncols_B = size(B, 2)
    num_steps = length(ts) - 1  # q
    t_min, t_max = extrema(ts)
    t_span = t_max - t_min
    Δt = step(ts)

    if shift
        μ = tr(A) / n
        A -= μ * I
    else
        μ = zero(float(eltype(A)))
    end

    degree_opt, scale = parameters(t_span, A, ncols_B; tol)  # (m*, s)

    F = F1 = expv(t_min, A, B; shift=false, tol) * exp(μ * t_min)

    if num_steps <= scale
        η = exp(Δt * μ)
        Fs_tail = map(1:num_steps) do k
            F = expv_taylor(k * Δt, A, F, degree_opt; tol) * η
            return F
        end
        Fs = vcat([F1], Fs_tail)
        return Fs
    end

    num_steps_per_scale = fld(num_steps, scale)  # d
    scale_actual, num_steps_last = fldmod(num_steps, num_steps_per_scale)  # (j, r)

    Z = F
    l = 1
    Fs_tails = map(1:(scale_actual + 1)) do i
        if i > scale_actual
            num_steps_per_scale = num_steps_last
        end
        Zs = [Z]  # cache of matrix products
        Fs_tail = map(1:num_steps_per_scale) do k
            η = exp(μ * k * Δt)
            F, Zs = expv_taylor_cache(Δt, A, Z, degree_opt, k, Zs; tol)
            F *= η
            return F
        end
        Z = F
        return Fs_tail
    end
    Fs = reduce(vcat, [[F1], Fs_tails...])
    return Fs
end
