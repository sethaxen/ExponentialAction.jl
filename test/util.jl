using ExponentialAction, LinearAlgebra

@testset "Utilities" begin
    @testset "_opnormInf" begin
        A = randn(ComplexF64, 10)
        @test ExponentialAction._opnormInf(A) ≈
              ExponentialAction._opnormInf(reshape(A, 10, 1))
        @test ExponentialAction._opnormInf(A) ≈ norm(A, Inf)
        A = randn(ComplexF64, 10, 10)
        @test ExponentialAction._opnormInf(A) ≈ opnorm(A, Inf)
    end

    @testset "opnormest1" begin
        A = randn(ComplexF64, 10, 10)
        @test ExponentialAction.opnormest1(A) ≈ opnorm(A, 1)
    end
end
