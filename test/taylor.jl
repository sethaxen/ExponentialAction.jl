using ExponentialAction, Test

@testset "expv_taylor" begin
    for i in 1:0.1:10
        t = rand() * i
        A = randn(5, 5)
        A /= opnorm(A, 1)
        B = randn(5, 2)
        @test ExponentialAction.expv_taylor(t, A, B, 1_000) ≈ exp(t * A) * B
    end
end
