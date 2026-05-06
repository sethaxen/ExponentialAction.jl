## Operator (matrix) norm estimation methods

"""
    $(TYPEDEF)

Abstract type for methods that estimate some operator norm of an operator.

Any method of type `M` must implement the following interface:
```julia
allocate_memory(method::M, operator, prototype)::C            # allocates the cache
opnorm_est(rng, method::M, operator, cache::C) -> opnorm_est  # overwrites the cache
```
Where `C` is some cache type.
`prototype` can be either `nothing` or an array such that
`mul!(::AbstractArray, operator, prototype)` is a valid operation.
It is used by `similar` to allocate the cache, which allows for GPU allocation of the cache.
"""
abstract type OperatorNormEstimationMethod end

"""
    HighamTisseurOpNorm1(ncols=2, maxiter=5)

The operator 1-norm estimation algorithm of Higham and Tisseur [^HighamTisseur2000].

The resulting estimate is a lower bound on the actual operator norm. and typically increases
with `ncols`.
The algorithm performs on average `4*ncols` matrix-vector products before convergence.

$(TYPEDFIELDS)

# References

[^HighamTisseur2000]: Higham NJ and Tisseur F. "A block algorithm for matrix 1-norm estimation, with an application to 1-norm pseudospectra."
                      SIAM J Matrix Anal. 21.4 (2000): 1185-1201.
                      doi: [10.1137/S0895479899356080](https://doi.org/10.1137/S0895479899356080)
                      [eprint](https://eprints.maths.manchester.ac.uk/id/eprint/321)
"""
@kwdef struct HighamTisseurOpNorm1 <: OperatorNormEstimationMethod
    "Number of columns of the generated test matrix that the operator will be applied to."
    ncols::Int = 2
    "Maximum number of iterations to run the algorithm. Must be at least 2."
    maxiter::Int = 5
    function HighamTisseurOpNorm1(ncols::Int, maxiter::Int)
        ncols ≥ 1 || throw(ArgumentError("ncols must be at least 1"))
        maxiter ≥ 2 || throw(ArgumentError("maxiter must be at least 2"))
        return new(ncols, maxiter)
    end
end
HighamTisseurOpNorm1(ncols::Int) = HighamTisseurOpNorm1(; ncols)

num_matvecs(method::HighamTisseurOpNorm1) = 4 * method.ncols

default_opnorm_method(t, A, B) = HighamTisseurOpNorm1(; prototype = B)

struct HighamTisseurOpNorm1Cache{T, Tr, C, R, I}
    "Test matrix to apply the operator to."
    X::Tr
    "Result of applying the operator to `X`."
    Y1::T
    "Result of applying the operator to `X`."
    Y2::T
    "Container for norms of columns of `Y1` or `Y2`."
    col_abs_sum::C
    "Container for norms of rows of `Y1` or `Y2`."
    row_abs_sum::R
    "Permutation of indices of `row_abs_sum`."
    ind::I
    "History of indices of `ind`."
    ind_hist::Set{Int}
end

function allocate_memory(method::HighamTisseurOpNorm1, A, prototype)
    (; ncols, maxiter) = method
    n = size(A, 2)
    ncols = min(n, ncols)

    ax_col = _build_axes(A, prototype, ncols)
    ax_row = _build_axes(A, prototype)

    # allocate all memory
    T = typeof(one(float(eltype(A))))
    rT = real(T)
    X = _similar_to_prototype(prototype, rT, (ax_row, ax_col))
    Y1 = similar(X, T)
    Y2 = similar(X, T)
    col_abs_sum = similar(X, (ax_col,))
    row_abs_sum = similar(X, (ax_row,))
    ind = similar(row_abs_sum, Int)
    ind_hist = Set{Int}()
    sizehint!(ind_hist, maxiter * ncols)

    cache = HighamTisseurOpNorm1Cache(X, Y1, Y2, col_abs_sum, row_abs_sum, ind, ind_hist)

    return cache
end

function opnorm_est(rng, method::HighamTisseurOpNorm1, A, cache::HighamTisseurOpNorm1Cache)
    # destructure everything and define aliases for convenience
    (; maxiter) = method
    (; X, Y1, Y2, col_abs_sum, row_abs_sum, ind, ind_hist) = cache
    Y, S_old, absY, S_dot, X_dot = Y1, Y2, X, col_abs_sum, col_abs_sum
    ncols = size(X, 2)
    row_ax = axes(X, 1)

    # compute the estimate
    iter = 0
    est = est_old = zero(eltype(col_abs_sum))
    empty!(ind_hist)
    _init_starting_matrix!(rng, X, X_dot)
    idx_prefix = first(row_ax, ncols)
    ind_prefix = view(ind, idx_prefix)
    idx_offset = first(row_ax) - 1  # cols are 1-indexed, but rows may not be
    while true
        iter += 1
        LinearAlgebra.mul!(Y, A, X)
        absY .= abs.(Y)
        sum!(col_abs_sum', absY)
        est, est_ind = findmax(col_abs_sum)
        if est > est_old || iter == 2
            ind_best = ArrayInterface.allowed_getindex(ind, idx_offset + est_ind)
        end
        # (1)
        if iter > 1 && est ≤ est_old
            est = est_old
            break
        end
        iter > maxiter && break
        est_old = est
        S = Y
        S .= ifelse.(iszero.(absY), one(eltype(Y)), Y ./ absY)
        if eltype(S) <: Real  # parallel cols unlikely for complex arrays, so only check for real
            # (2) algorithm about to converge, so skip the next mul!(S, A', S) and the next maximum!
            iter > 1 && _each_col_has_parallel_col!(S_dot, S, S_old) && break
            ncols > 1 && _resample_parallel_cols!(rng, S, iter > 1 ? S_old : nothing, S_dot)
        end
        # (3)
        Y, S_old = S_old, S
        LinearAlgebra.mul!(Y, A', S)
        absY .= abs.(Y)
        maximum!(row_abs_sum, absY)
        # (4)
        iter > 1 && maximum(row_abs_sum) == ArrayInterface.allowed_getindex(row_abs_sum, ind_best) && break
        sortperm!(ind, row_abs_sum; rev = true)
        if ncols > 1
            # (5)
            issubset(ind_prefix, ind_hist) && break
            # replace first ncols entries in ind with first ncols entries not in ind_hist, if possible
            partialsort!(ind, idx_prefix; by = ∈(ind_hist))
        end
        copyto!(view(X, ind, :), I)
        union!(ind_hist, ind_prefix)
    end
    return est
end

# NOTE: the paper describes that for complex operators, the starting matrix should be real
# and is initialized with entries of unit norm scaled by 1/n, just like for real operators.
# The matlab implementation of normest1 does the same.
# However, what seems to be the reference LAPACK implementation for complex operators ZLACN1
# (described in LAPACK working note 152 https://www.netlib.org/lapack/lawnspdf/lawn152.pdf
# and archived from http://www.cs.man.ac.uk/~scheng/PCMF/zlacn1.tar on Jan 9, 2002) uses
# a complex starting matrix with random *complex* entries of unit norm.
# We use the same approach as the paper and matlab implementation.
function _init_starting_matrix!(rng::Random.AbstractRNG, X, X_dot)
    ncols = size(X, 2)
    fill!(view(X, :, 1), 1)
    _rand_signs!(rng, view(X, :, 2:ncols))
    _resample_parallel_cols!(rng, X, nothing, X_dot)
    X ./= size(X, 1)
    return X
end

function _resample_parallel_cols!(rng, X, X_old, X_dot)
    n, ncols = size(X)
    # Limit to maxiter mat-vecs. Since each mat-vec scales as O(n*ncols), and we need to check O(ncols),
    # this limit keeps the total resampling cost to O(n²*ncols), same as the rest of the algorithm,
    # and prevents it from dominating the overall cost.
    # Cheng SH, Higham NJ. (2001). Parallel Implementation of a Block Algorithm for Matrix 1-Norm Estimation.
    # Euro-Par 2001. doi: 10.1007/3-540-44681-8_82
    maxiter = fld(n, ncols)
    for (j, Xj) in enumerate(eachcol(X))
        iter = 0
        while iter < maxiter
            has_parallel = false
            if j > 1
                X_dot_view = view(X_dot, 1:(j - 1))
                LinearAlgebra.mul!(X_dot_view, view(X, :, 1:(j - 1))', Xj)
                iter += 1
                has_parallel = round(Int, maximum(abs, X_dot_view)) == n
            end
            if X_old !== nothing && !has_parallel
                LinearAlgebra.mul!(X_dot, X_old', Xj)
                iter += 1
                has_parallel = round(Int, maximum(abs, X_dot)) == n
            end
            has_parallel || break
            _rand_signs!(rng, Xj)
        end
    end
    return
end

function _each_col_has_parallel_col!(X_dot, X, X_old)
    n = size(X, 1)
    rt = all(eachcol(X)) do Xj
        LinearAlgebra.mul!(X_dot, X_old', Xj)
        round(Int, maximum(abs, X_dot)) == n
    end
    return rt
end
