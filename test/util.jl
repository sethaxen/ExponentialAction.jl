using ExponentialAction, ChainRulesCore, LinearAlgebra, Test

@testset "Utilities" begin
    @testset "_opnormInf" begin
        A = randn(ComplexF64, 10)
        @test ExponentialAction._opnormInf(A) ≈
              ExponentialAction._opnormInf(reshape(A, 10, 1))
        @test ExponentialAction._opnormInf(A) ≈ norm(A, Inf)
        A = randn(ComplexF64, 10, 10)
        @test ExponentialAction._opnormInf(A) ≈ opnorm(A, Inf)
    end

    @testset "_opnormInf non-differentiable" begin
        A = randn(ComplexF64, 10)
        n, back = ChainRulesCore.rrule(ExponentialAction._opnormInf, A)
        @test n == ExponentialAction._opnormInf(A)
        @test @inferred(back(1.0)) === (DoesNotExist(), DoesNotExist())
        A = randn(ComplexF64, 10, 10)
        n, back = ChainRulesCore.rrule(ExponentialAction._opnormInf, A)
        @test n == ExponentialAction._opnormInf(A)
        @test @inferred(back(1.0)) === (DoesNotExist(), DoesNotExist())
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
