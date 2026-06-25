using ExponentialAction
using LinearAlgebra
using Random
using Statistics
using Test

@testset "trace_est" begin
    @testset "ExplicitTrace" begin
        @testset for T in (Int, Float32, Float64, ComplexF32, ComplexF64), n in (1, 5, 20)
            A = T <: Int ? rand(1:5, n, n) : randn(T, n, n)
            method = ExponentialAction.ExplicitTrace()
            rng = Xoshiro(42)
            trA = tr(A)
            tr_est, tr_se = @inferred ExponentialAction.trace_est(rng, method, A)
            @test tr_est == trA
            @test iszero(tr_se)
            @test tr_est isa T
            prototype = zeros(7)
            tr_est, tr_se = @inferred ExponentialAction.trace_est(rng, method, A, prototype)
            @test tr_est == trA
            @test iszero(tr_se)
        end
    end

    @testset "XTrace" begin
        @testset "constructor" begin
            @testset for num_matvecs in (2, 4), resphered in (true, false)
                method = ExponentialAction.XTrace(num_matvecs; resphered)
                @test method.num_matvecs == num_matvecs
                @test method.resphered == resphered
            end
            method = ExponentialAction.XTrace(10)
            @test method.num_matvecs == 10
            @test method.resphered
        end

        @testset "reproducibility" begin
            A = randn(12, 12)
            method = ExponentialAction.XTrace(8)
            a = ExponentialAction.trace_est(Xoshiro(2025), method, A)
            b = ExponentialAction.trace_est(Xoshiro(2025), method, A)
            @test a == b
        end

        @testset "converges instantly on scalar matrix" begin
            @testset for T in (Float64, ComplexF64), n in (8, 32)
                α = randn(T) * 20
                A = α * I(n)
                trA = tr(A)
                method = ExponentialAction.XTrace(2)
                ests = [ExponentialAction.trace_est(Xoshiro(k), method, A)[1] for k in 1:200]
                @test all(Base.Fix2(isapprox, trA), ests)
            end
        end

        @testset "tr(A) within 95% interval" begin
            n = 40
            A = randn(n, n)
            trA = tr(A)
            method = ExponentialAction.XTrace(5)
            nreps = 8_000
            seeds = rand(UInt8, nreps)
            ests = map(seeds) do seed
                ExponentialAction.trace_est(Xoshiro(seed), method, A)[1]
            end
            α = 0.05
            lb = quantile(ests, α / 2)
            ub = quantile(ests, 1 - α / 2)
            @test lb ≤ trA ≤ ub
        end

        @testset "tr(A) within 95% region using standard error" begin
            n = 1_000
            nreps = 20
            @testset for T in (Float64, ComplexF64), matvecs in (20, 50, 100)
                seeds = rand(UInt8, nreps)
                method = ExponentialAction.XTrace(matvecs)
                @testset for i in 1:nreps
                    rng = Xoshiro(seeds[i])
                    A = randn(rng, T, n, n)
                    tr_est, tr_se = ExponentialAction.trace_est(rng, method, A)
                    @test tr_est isa T
                    @test tr_se isa real(T)
                    @test tr_se ≥ 0
                    @test isfinite(tr_se)
                    @test abs(tr(A) - tr_est) ≤ 3.22 * tr_se  # 95% confidence interval corrected for 2*nreps tests
                end
            end
        end

        @testset "Can accept a linear operator" begin
            n = 15
            α = 2.0
            A = Matrix(α * I, n, n)
            trA = tr(A)
            op = LinOp(A)
            method = ExponentialAction.XTrace(14)
            tr_est, tr_se = ExponentialAction.trace_est(Xoshiro(7), method, op)
            @test tr_est ≈ trA
            @test tr_se ≥ 0
        end

        @testset "consistent with experiments in paper" begin
            # NOTE: probably due to numerical differences between matlab and julia,
            # we typically observe lower mean relative errors than reported in the paper.
            n = 1_000
            nreps = 10  # 1000 in the paper, but 10 is good enough for our tests
            seeds = rand(UInt8, nreps)

            @testset "flat" begin
                λ = range(3, 1; length = n)
                rel_errs = [(20, false) => 4.0e-3, (300, false) => 2.0e-3, (300, true) => 1.0e-3]
                @testset "matvecs=$matvecs, resphered=$resphered" for ((matvecs, resphered), rel_err) in rel_errs
                    method = ExponentialAction.XTrace(matvecs; resphered)
                    mean_rel_err = mean(seeds) do seed
                        rng = Xoshiro(seed)
                        U = qr(randn(rng, n, n)).Q
                        A = U * Diagonal(λ) * U'
                        tr_est, _ = ExponentialAction.trace_est(rng, method, A)
                        return abs(tr(A) - tr_est) / abs(tr(A))
                    end
                    @test mean_rel_err < rel_err
                end
            end

            @testset "step" begin
                λ = [ones(50); fill(1.0e-3, n - 50)]
                rel_errs = [
                    (20, false) => 0.1, (125, false) => 1.0e-4, (200, false) => 1.0e-4,
                    (125, true) => 1.0e-5, (200, true) => 1.0e-6,
                ]
                @testset "matvecs=$matvecs, resphered=$resphered" for ((matvecs, resphered), rel_err) in rel_errs
                    method = ExponentialAction.XTrace(matvecs; resphered)
                    mean_rel_err = mean(seeds) do seed
                        rng = Xoshiro(seed)
                        U = qr(randn(rng, n, n)).Q
                        A = U * Diagonal(λ) * U'
                        tr_est, _ = ExponentialAction.trace_est(rng, method, A)
                        return abs(tr(A) - tr_est) / abs(tr(A))
                    end
                    @test mean_rel_err < rel_err
                end
            end

            @testset "exp" begin
                λ = 0.7 .^ (0:(n - 1))
                rel_errs = [20 => 0.1, 100 => 1.0e-6, 200 => 1.0e-14]
                @testset "matvecs=$matvecs" for (matvecs, rel_err) in rel_errs
                    method = ExponentialAction.XTrace(matvecs; resphered = false)
                    mean_rel_err = mean(seeds) do seed
                        rng = Xoshiro(seed)
                        U = qr(randn(rng, n, n)).Q
                        A = U * Diagonal(λ) * U'
                        tr_est, _ = ExponentialAction.trace_est(rng, method, A)
                        return abs(tr(A) - tr_est) / abs(tr(A))
                    end
                    @test mean_rel_err < rel_err
                end
            end
        end
    end
end
