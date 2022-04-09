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

This algorithm takes special care to avoid overscaling and is described in Code Fragment 5.1
of [^AlMohyHigham2011].
"""
function expv_sequence(ts, A, B; kwargs...)
    F = expv(ts[begin], A, B; kwargs...)
    Fs = similar(ts, typeof(F))
    Fs[begin] = F
    for i in eachindex(ts)[(begin + 1):end]
        Δt = @inbounds ts[i] - ts[i - 1]
        F = expv(Δt, A, F; kwargs...)
        Fs[i] = F
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

    F = expv(t_min, A, B; shift=false, tol) * exp(μ * t_min)
    Fs = Vector{typeof(F)}(undef, num_steps + 1)
    Fs[1] = F

    degree_opt, scale = parameters(t_span, A, ncols_B; tol)  # (m*, s)
    num_steps_per_scale = fld(num_steps, scale)  # d
    scale_actual, num_steps_last = fldmod(num_steps, num_steps_per_scale)  # (j, r)

    Z = F
    l = 1
    for i in 1:scale_actual
        if i > scale_actual
            num_steps_per_scale = num_steps_last
        end
        for k in 1:num_steps_per_scale
            l += 1
            η = exp(μ * k * Δt)
            Fs[l] = F = expv_taylor(k * Δt, A, Z, degree_opt; tol) * η
        end
        Z = F
    end

    return Fs
end
