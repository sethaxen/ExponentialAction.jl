using CUDA
using ExponentialAction
using LinearAlgebra
using Random
using Test

if !CUDA.has_cuda()
    @warn "CUDA not found - skipping GPU tests"
end

CUDA.allowscalar(false)

CUDA.has_cuda() && @testset "CUDA integration" begin
    struct CumsumLinOp{T <: Real}
        n::Int
        CumsumLinOp(::Type{T}, n::Int) where {T <: Real} = new{T}(n)
    end
    LinearAlgebra.size(op::CumsumLinOp, i) = (op.n, op.n)[i]
    Base.eltype(::CumsumLinOp{T}) where {T} = T
    function LinearAlgebra.mul!(Y, ::CumsumLinOp, X)
        cumsum!(Y, X; dims = 1)
        return Y
    end
    Base.adjoint(op::CumsumLinOp) = AdjointCumsumLinOp(eltype(op), op.n)

    struct AdjointCumsumLinOp{T <: Real}
        n::Int
        AdjointCumsumLinOp(::Type{T}, n::Int) where {T <: Real} = new{T}(n)
    end
    LinearAlgebra.size(op::AdjointCumsumLinOp, i) = (op.n, op.n)[i]
    Base.eltype(::AdjointCumsumLinOp{T}) where {T} = T
    function LinearAlgebra.mul!(Y, ::AdjointCumsumLinOp, X)
        cumsum!(Y, reverse(X; dims = 1); dims = 1)
        reverse!(Y; dims = 1)
        return Y
    end
    Base.adjoint(op::AdjointCumsumLinOp) = CumsumLinOp(eltype(op), op.n)

    _all_cache_arrays_on_gpu(cache) = all(
        x -> x isa CuArray,
        (cache.X, cache.Y1, cache.Y2, cache.col_abs_sum, cache.row_abs_sum, cache.ind, cache.seen, cache.ind_rank),
    )

    @testset "GPU opnorm_est" begin
        @testset "cache inference and allocation backend" begin
            n = 16
            A = CUDA.rand(Float32, n, n)
            method = ExponentialAction.HighamTisseurOpNorm1(; ncols = 3, maxiter = 4)

            v_prototype = CUDA.zeros(Float32, n)
            cache_vec = @inferred ExponentialAction.allocate_memory(method, A, v_prototype)
            @test _all_cache_arrays_on_gpu(cache_vec)

            m_prototype = CUDA.zeros(Float32, n, 2)
            cache_mat = @inferred ExponentialAction.allocate_memory(method, A, m_prototype)
            @test _all_cache_arrays_on_gpu(cache_mat)
        end

        @testset "opnorm_est executes with CUDA-compatible custom linop" begin
            n = 20
            rng = Xoshiro(7)
            op = CumsumLinOp(Float32, n)
            method = ExponentialAction.HighamTisseurOpNorm1(; ncols = 2, maxiter = 4)
            cache = ExponentialAction.allocate_memory(method, op, CUDA.zeros(Float32, n))
            est = ExponentialAction.opnorm_est(rng, method, op, cache)
            @test est isa Float32
            @test isfinite(est)
            @test est ≥ 0
        end

        @testset "utility methods execute on CuArrays" begin
            n, ncols = 6, 4
            # rng = Xoshiro(13)

            @testset "init_starting_matrix!" begin
                rng = cuRAND.LibraryRNG()
                X = CUDA.zeros(Float32, n, ncols)
                X_dot = CUDA.zeros(Float32, ncols)
                ExponentialAction._init_starting_matrix!(rng, X, X_dot)
                # can't be strictly non-allocating because mul! isn't, but should be low
                @test CUDA.@allocated(ExponentialAction._init_starting_matrix!(rng, X, X_dot)) < 100
                for _ in 1:1_000
                    fill!(X, 0)
                    fill!(X_dot, 0)
                    ExponentialAction._init_starting_matrix!(rng, X, X_dot)
                    @test all(isapprox.(sum(abs.(X); dims = 1), 1))
                    @test all(isapprox.(X[:, 1], 1.0f0 / n))
                    @test !any(isapprox.(triu(X'X, 1), 1))
                end
            end

            @testset "resample_parallel_cols!" begin
                rng = Xoshiro(90)
                X_dot = CUDA.zeros(Float32, ncols)

                X_parallel = CUDA.fill(1.0f0, n, ncols)
                ExponentialAction._resample_parallel_cols!(rng, X_parallel, nothing, X_dot)
                @test all(sum(abs.(X_parallel); dims = 1) .≈ n)

                X_old = CUDA.fill(1.0f0, n, ncols)
                ExponentialAction._resample_parallel_cols!(rng, X_parallel, X_old, X_dot)
                @test all(sum(abs.(X_parallel); dims = 1) .≈ n)

                X_parallel = CUDA.fill(1.0f0, n, ncols)
                # can't be strictly non-allocating because mul! isn't, but should be low
                @test CUDA.@allocated(ExponentialAction._resample_parallel_cols!(rng, X_parallel, X_old, X_dot)) < 100
            end

            @testset "each_col_has_parallel_col!" begin
                X_old = CUDA.fill(1.0f0, n, ncols)
                X_dot = CUDA.zeros(Float32, ncols)
                X_same = CUDA.fill(1.0f0, n, ncols)
                @test ExponentialAction._each_col_has_parallel_col!(X_dot, X_same, X_old)

                X_diff = CUDA.fill(1.0f0, n, ncols)
                view(X_diff, 1:1, 1:1) .= -1.0f0
                @test !ExponentialAction._each_col_has_parallel_col!(X_dot, X_diff, X_old)
            end
        end
    end
end
