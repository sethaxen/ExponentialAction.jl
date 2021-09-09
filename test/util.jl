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

    @testset "_cld" begin
        @test @inferred(ExponentialAction._cld(1, 2)) === 1
        @test @inferred(ExponentialAction._cld(6, 2.5)) === 3
        @test @inferred(ExponentialAction._cld(6, 3)) === 2
        @test @inferred(ExponentialAction._cld(6, 4)) === 2
        @test @inferred(ExponentialAction._cld(1, 0)) === typemax(Int)
    end
end
