using ExponentialAction
using LinearAlgebra
using Random
using SparseArrays
using Statistics
using Test

function run_experiments(rng, methods, build_op, nreps)
    A = build_op()
    caches = ExponentialAction.allocate_memory.(methods, Ref(A), nothing)
    rngs = [deepcopy(rng) for _ in methods]
    results = stack(
        map(1:nreps) do _
            A = build_op()
            norm = opnorm(A, 1)
            map(rngs, methods, caches) do rng, method, cache
                op = MatVecCountingLinOp(A)
                norm_est = ExponentialAction.opnorm_est(rng, method, op, cache)
                ratio = norm_est / norm
                nprod = op.matvecs[] ÷ method.ncols
                return (; ratio, nprod)
            end
        end
    )
    ratios = map(r -> r.ratio, results)
    nprods = map(r -> r.nprod, results)
    return (; ratios, nprods)
end

@testset "opnorm_est" begin
    @testset "HighamTisseurOpNorm1" begin
        @testset "constructor and default values" begin
            method = ExponentialAction.HighamTisseurOpNorm1()
            @test method.ncols == 2
            @test method.maxiter == 5
            @testset for (ncols, maxiter) in [(1, 2), (2, 3), (3, 4)]
                method = ExponentialAction.HighamTisseurOpNorm1(; ncols, maxiter)
                @test method.ncols == ncols
                @test method.maxiter == maxiter
                method = ExponentialAction.HighamTisseurOpNorm1(ncols, maxiter)
                @test method.ncols == ncols
                @test method.maxiter == maxiter
                method = ExponentialAction.HighamTisseurOpNorm1(ncols)
                @test method.ncols == ncols
                @test method.maxiter == 5
            end
        end

        @testset "argument validation" begin
            @test_throws ArgumentError ExponentialAction.HighamTisseurOpNorm1(; ncols = 0)
            @test_throws ArgumentError ExponentialAction.HighamTisseurOpNorm1(; maxiter = 1)
        end

        @testset "cache allocation" begin
            @testset for n in (5, 10), T in (Int, Float32, Float64, ComplexF64), Tproto in (Nothing, Array)
                A = Array{T}(undef, n, n)
                prototype = Tproto === Nothing ? nothing : Array{Int}(undef, n)
                Tf = float(T)
                Tr = real(Tf)
                @testset for ncols in (3, 5, n + 1)
                    ncols_eff = min(ncols, n)
                    method = ExponentialAction.HighamTisseurOpNorm1(; ncols)
                    cache = @inferred ExponentialAction.allocate_memory(method, A, prototype)
                    @test cache.X isa Matrix{Tr}
                    @test size(cache.X) == (n, ncols_eff)
                    @test cache.Y1 isa Matrix{Tf}
                    @test size(cache.Y1) == (n, ncols_eff)
                    @test cache.Y2 isa Matrix{Tf}
                    @test size(cache.Y2) == (n, ncols_eff)
                    @test cache.col_abs_sum isa Vector{Tr}
                    @test size(cache.col_abs_sum) == (ncols_eff,)
                    @test cache.row_abs_sum isa Vector{Tr}
                    @test size(cache.row_abs_sum) == (n,)
                    @test cache.ind isa Vector{Int}
                    @test size(cache.ind) == (n,)
                    @test cache.seen isa Vector{Bool}
                    @test size(cache.seen) == (n,)
                    @test cache.ind_rank isa Vector{Int}
                    @test size(cache.ind_rank) == (n,)
                end
            end
        end

        @testset "non-allocating" begin
            @testset for n in (5, 10), T in (Int, Float32, Float64, ComplexF64), ncols in (2, 3, 5), maxiter in (2, 3, 4)
                A = T <: Int ? rand((-1, 0, 1), n, n) : randn(T, n, n)
                method = ExponentialAction.HighamTisseurOpNorm1(; ncols, maxiter)
                cache = ExponentialAction.allocate_memory(method, A, nothing)
                cache_copy = deepcopy(cache)
                rng = Xoshiro(42)
                ExponentialAction.opnorm_est(rng, method, A, cache)  # make sure it's precompiled
                @test @allocated(ExponentialAction.opnorm_est(rng, method, A, cache)) == 0
            end
        end

        @testset "basic properties" begin
            @testset for T in (Float32, Float64, ComplexF32, ComplexF64), TA in (Matrix, Bidiagonal, Tridiagonal, LinOp), n in (10, 100)
                A = TA <: Bidiagonal ? TA(randn(T, n, n), :U) : TA(randn(T, n, n))
                prototype = randn(n)
                method = ExponentialAction.HighamTisseurOpNorm1()
                cache = @inferred ExponentialAction.allocate_memory(method, A, prototype)
                norm_est = @inferred ExponentialAction.opnorm_est(Xoshiro(42), method, A, cache)
                @test norm_est isa real(T)
                norm = A isa LinOp ? opnorm(A.A, 1) : opnorm(A, 1)
                @test norm_est ≤ norm  # estimate is a lower bound on the true norm
                @test norm_est ≈ norm rtol = 0.5  # usually norm_est/norm >= 0.5
            end

        end
        @testset "produces expected results for specific matrices" begin
            @testset "exact norm computed after 2 iterations for positive and rademacher matrices" begin
                @testset for n in (10, 100), dist in (:u1, :u100, :rademacher)
                    A = if dist === :u1
                        rand(n, n)
                    elseif dist === :u100
                        rand(n, n) * 100
                    elseif dist === :rademacher
                        rand((-1.0, 1.0), n, n)
                    end
                    method = ExponentialAction.HighamTisseurOpNorm1(; ncols = 1, maxiter = 2)
                    cache = ExponentialAction.allocate_memory(method, A, nothing)
                    norm_est = ExponentialAction.opnorm_est(Xoshiro(98), method, A, cache)
                    norm = opnorm(A, 1)
                    @test norm_est ≈ norm
                end
            end

            @testset "parallel cols detected and resampled" begin
                # empirically, this matrix triggers the parallel cols detection and resampling for most seeds
                #! format: off
                A = Float64[
                     6  -9   5  -5
                    -1   1   9   2
                    -9  -7  -4  -4
                     6  -9  -9  -4
                ]
                #! format: on
                norm = opnorm(A, 1)
                method = ExponentialAction.HighamTisseurOpNorm1(; ncols = 2, maxiter = 5)
                cache = ExponentialAction.allocate_memory(method, A, nothing)
                rng = Xoshiro(38)
                frac_exact = mean(1:1000) do _
                    norm_est = ExponentialAction.opnorm_est(rng, method, A, cache)
                    norm_est ≈ norm
                end
                @test frac_exact > 0.6  # empirically, ~ 0.64 with resampling, ~ 0.57 without
            end
        end

        @testset "inverse of random normal matrix" begin
            n = 100
            nreps = 5_000
            A = Array{Float64}(undef, n, n)

            # Table 3 from paper
            ncols_values = [2, 4, 6]
            ratios_expected = [0.993, 0.999, 1.0]
            max_nprods_expected = [8, 6, 6]
            mean_nprods_expected = fill(4, 3)

            methods = map(ExponentialAction.HighamTisseurOpNorm1, ncols_values)
            results = run_experiments(Xoshiro(42), methods, () -> inv(randn!(A)), nreps)
            @test vec(mean(results.ratios; dims = 2)) ≈ ratios_expected atol = 0.003
            @test vec(mean(results.nprods; dims = 2)) ≈ mean_nprods_expected atol = 0.3
            @test all(vec(maximum(results.nprods; dims = 2)) .≤ max_nprods_expected)
        end

        @testset "random matrix with values in (-1, 0, 1)" begin
            n = 100
            nreps = 5_000
            A = Array{Float64}(undef, n, n)

            # Table 4 from paper
            ncols_values = [2, 6, 10]
            ratios_expected = [0.883, 0.935, 0.956]
            max_nprods_expected = [4, 4, 4]
            mean_nprods_expected = fill(4, 3)

            methods = map(ExponentialAction.HighamTisseurOpNorm1, ncols_values)
            results = run_experiments(Xoshiro(42), methods, () -> rand!(A, -1.0:1.0), nreps)

            @test vec(mean(results.ratios; dims = 2)) ≈ ratios_expected atol = 0.03
            @test vec(mean(results.nprods; dims = 2)) ≈ mean_nprods_expected atol = 0.3
            @test all(vec(maximum(results.nprods; dims = 2)) .≤ max_nprods_expected)
        end

        @testset "inverse of random uniform complex matrix" begin
            n = 100
            nreps = 5_000
            A = Array{ComplexF64}(undef, n, n)

            # Table 6 from paper
            ncols_values = [1, 2, 4, 6]
            ratios_expected = [0.98, 0.994, 0.999, 1.0]
            max_nprods_expected = [8, 8, 6, 6]
            mean_nprods_expected = fill(4, 4)

            methods = map(ExponentialAction.HighamTisseurOpNorm1, ncols_values)
            results = run_experiments(Xoshiro(42), methods, () -> inv(rand!(A)), nreps)
            @test vec(mean(results.ratios; dims = 2)) ≈ ratios_expected atol = 0.004
            @test vec(mean(results.nprods; dims = 2)) ≈ mean_nprods_expected atol = 0.4
            @test all(vec(maximum(results.nprods; dims = 2)) .≤ max_nprods_expected)
        end
    end
end
