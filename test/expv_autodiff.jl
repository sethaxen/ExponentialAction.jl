using ExponentialAction
using FiniteDifferences
using ForwardDiff
using ReverseDiff
using Zygote

@testset "expv automatic differentiation" begin
    @testset "ForwardDiff" begin
        t = rand()
        A = randn(10, 10)
        B = randn(10)
        tjac_fd, Ajac_fd, Bjac_fd = FiniteDifferences.jacobian(central_fdm(5, 1), expv, t, A, B)
        tjac_ad = ForwardDiff.derivative(t -> expv(t, A, B), t)
        @test tjac_ad ≈ tjac_fd
        Ajac_ad = ForwardDiff.jacobian(A -> expv(t, A, B), A)
        @test Ajac_fd ≈ Ajac_fd
        Bjac_ad = ForwardDiff.jacobian(B -> expv(t, A, B), B)
        @test Bjac_fd ≈ Bjac_fd
    end

    @testset "ReverseDiff" begin
        t = rand()
        A = randn(10, 10)
        B = randn(10)
        tjac_fd, Ajac_fd, Bjac_fd = FiniteDifferences.jacobian(central_fdm(5, 1), expv, t, A, B)
        tjac_ad = vec(ReverseDiff.jacobian(t -> expv(first(t), A, B), [t]))
        @test tjac_ad ≈ tjac_fd
        Ajac_ad = ReverseDiff.jacobian(A -> expv(t, A, B), A)
        @test Ajac_fd ≈ Ajac_fd
        Bjac_ad = ReverseDiff.jacobian(B -> expv(t, A, B), B)
        @test Bjac_fd ≈ Bjac_fd
    end

    @testset "Zygote" begin
        @testset for T in (Float64, ComplexF64)
            t = rand(T)
            A = randn(T, 10, 10)
            B = randn(T, 10)
            Ȳ = randn(T, 10)
            t̄_fd, Ā_fd, B̄_fd = j′vp(central_fdm(5, 1), expv, Ȳ, t, A, B)
            Y, back = Zygote.pullback(expv, t, A, B)
            @test Y ≈ expv(t, A, B)
            t̄_ad, Ā_ad, B̄_ad = back(Ȳ)
            @test t̄_ad ≈ t̄_fd
            @test Ā_ad ≈ Ā_fd
            @test B̄_ad ≈ B̄_fd
        end
    end
end
