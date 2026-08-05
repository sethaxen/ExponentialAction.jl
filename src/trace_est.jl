"""
    TraceEstimationMethod

Abstract type for methods that estimate the trace of an operator.

All methods must implement the following interface:
```julia
trace_est(rng, method::TraceEstimationMethod, operator, prototype=nothing) -> (tr_est, tr_se)
```
`prototype` can be either `nothing` or an array such that
`mul!(::AbstractArray, operator, prototype)` is a valid operation.
If provided, it is used by `similar` to allocate any intermediate arrays.
"""
abstract type TraceEstimationMethod end

"""
    ExplicitTrace()

Compute the trace of an operator using `LinearAlgebra.tr`.
"""
struct ExplicitTrace <: TraceEstimationMethod end

"""
    XTrace(num_matvecs::Int; resphered::Bool = true)

Compute an exchangeable trace estimate of of an operator using at most `num_matvecs` matrix-vector products.[^XTrace]

If `resphered` is `true`, the variance of the estimator is reduced by "resphering" the test vectors (see §13.1 of [^Epperly2025]).
This typically improves the accuracy when the operator exhibits slow spectral decay without significant trade-offs for operators
with fast spectral decay.

# References

[^XTrace]: Epperly EN, Tropp JA, Webber RJ (2024). Xtrace: Making the most of every sample in stochastic trace estimation.
           SIAM J Matrix Anal A. 45.1: 1-23.
           doi: [10.1137/23M1548323](https://doi.org/10.1137/23M1548323)
           arXiv: [2301.07825](https://arxiv.org/abs/2301.07825)
[^Epperly2025]: Epperly EN (2025). Make the most of what you have: Resource-efficient randomized algorithms for matrix computations. PhD Thesis.
                arXiv: [2512.15929](https://arxiv.org/abs/2512.15929)
"""
@kwdef struct XTrace <: TraceEstimationMethod
    num_matvecs::Int
    resphered::Bool = true
end
XTrace(num_matvecs::Int; kwargs...) = XTrace(; num_matvecs, kwargs...)

"""
    trace_est(rng, method::TraceEstimationMethod, A) -> (trA_est, trA_se)

Estimate the trace of `A` and its standard error using the specified `method`.

If random test vectors are sampled, `rng` should be provided.
"""
trace_est

function trace_est(rng, ::ExplicitTrace, A, prototype = nothing)
    trA = LinearAlgebra.tr(A)
    return trA, float(zero(trA))
end
function trace_est(rng, method::XTrace, A, prototype = nothing)
    (; num_matvecs, resphered) = method
    n = size(A, 2)
    nvecs = max(1, min(num_matvecs, n) ÷ 2)
    Ty = typeof(oneunit(float(eltype(A))))
    rTy = real(Ty)
    ax_row = _build_axes(A, prototype)
    ax_col = _build_axes(A, prototype, nvecs)

    # Allocate all large arrays (with n * nvecs elements)
    Y = Z = _similar_to_prototype(prototype, Ty, (ax_row, ax_col))
    Ω = similar(Y)
    Qmat = similar(Y)

    # sample test vectors
    if resphered
        _rand_sphere!(rng, Ω)
        LinearAlgebra.rmul!(Ω, sqrt(rTy(n)))
    else
        _rand_signs!(rng, Ω)
    end

    LinearAlgebra.mul!(Y, A, Ω)
    Q, R = LinearAlgebra.qr!(Y)
    copyto!(Qmat, Q)
    S = inv(LinearAlgebra.UpperTriangular(R)')
    LinearAlgebra.mul!(Z, A, Qmat)

    H = Qmat' * Z   # low-rank approximation of A using all vectors in Ω
    trH = LinearAlgebra.tr(H)
    W = Qmat' * Ω
    T = Z' * Ω

    tr_loos = map(eachcol(S), eachcol(W), eachcol(R), eachcol(T)) do s, w, r, t
        ss = sum(abs2, s)
        sw = LinearAlgebra.dot(s, w)
        sw_over_ss = sw / ss

        scale = if resphered
            (n - nvecs + 1) / (n - sum(abs2, w) + abs2(sw) / ss)
        else
            one(rTy)
        end

        x = LinearAlgebra.axpy!(-sw_over_ss, s, w)
        # trace of low-rank approximation Hᵢ ≈ A using Ω[:, Not(i)]
        trH_loo = trH - LinearAlgebra.dot(s, H, s) / ss
        # estimate of tr(A - Hᵢ) using Ω[:, i]
        resid_est = (
            -LinearAlgebra.dot(t, x) +
                LinearAlgebra.dot(x, H, x) +
                LinearAlgebra.dot(s, r) * conj(sw_over_ss)
        )

        return trH_loo + scale * resid_est
    end

    tr_est = Statistics.mean(tr_loos)
    tr_se = sqrt(Statistics.var(tr_loos; mean = tr_est, corrected = true) / nvecs)

    return tr_est, tr_se
end
