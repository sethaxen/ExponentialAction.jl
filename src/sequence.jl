"""
    expv_sequence(ts::AbstractVector, A, B; kwargs...)

Compute ``\\exp(t_i A)B`` for the (sorted) sequence of (real) time points ``\\{t_i\\}`.

At each time point, the result ``F_i`` is computed as
```math
F_i = \\exp{(t_i - t_{i-1}) A} F_{i - 1}
```
using [`expv`](@ref), where ``t_0 = 0`` and ``F_0 = B``.
For details, see Equation 5.2 of [^AlMohyHigham2011].

Because the cost of computing `expv` is related to the operator 1-norm of ``t_i A``, this
incremental computation is more efficient than computing `expv` separately for each time
point.

See [`expv`](@ref) for a description of acceptable `kwargs`.

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
