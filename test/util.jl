using ExponentialAction, ChainRulesCore, ChainRulesTestUtils, LinearAlgebra, Test
using ExponentialAction: _opnormInf

@testset "Utilities" begin
    @testset "_opnormInf" begin
        A = randn(ComplexF64, 10)
        @test _opnormInf(A) ≈ _opnormInf(reshape(A, 10, 1))
        @test _opnormInf(A) ≈ norm(A, Inf)
        A = randn(ComplexF64, 10, 10)
        @test _opnormInf(A) ≈ opnorm(A, Inf)
    end

    @testset "_opnormInf non-differentiable" begin
        test_rrule(_opnormInf, randn(ComplexF64, 10) ⊢ NoTangent(); atol=1e-6)
        test_rrule(_opnormInf, randn(ComplexF64, 10, 10) ⊢ NoTangent(); atol=1e-6)
    end

    @testset "opnormest1" begin
        A = randn(ComplexF64, 10, 10)
        @test ExponentialAction.opnormest1(A) ≈ opnorm(A, 1)
    end

    @testset "asint" begin
        @test ExponentialAction.asint(float(typemax(Int) * 10)) == typemax(Int)
        @test ExponentialAction.asint(37.0) == 37
    end

    @testset "default_tolerance" begin
        @test ExponentialAction.default_tolerance(randn()) ≈ eps(Float64)
        @test ExponentialAction.default_tolerance(randn(ComplexF64)) ≈ eps(Float64)
        @test ExponentialAction.default_tolerance(randn(Float32)) ≈ eps(Float32)
        @test ExponentialAction.default_tolerance(1, randn()) ≈ eps(Float64)
        test_rrule(ExponentialAction.default_tolerance, randn() ⊢ NoTangent(); atol=1e-6)
    end
end
