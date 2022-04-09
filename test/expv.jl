using ExponentialAction, LinearAlgebra, SparseArrays, Test

Tsingle = (Float32, ComplexF32)
Tdouble = (Float64, ComplexF64)

@testset "expv_taylor" begin
    for i in 1:0.1:10
        t = rand() * i
        A = randn(5, 5)
        A /= opnorm(A, 1)
        B = randn(5, 2)
        @test ExponentialAction.expv_taylor(t, A, B, 1_000) ≈ exp(t * A) * B
    end
end

@testset "expv" begin
    n = 10
    @testset "expv(t::$Tt, A::$MT{$TA}, B::Array{$TB,$(length(Bdims2)+1)}), tscale=$tscale, shift=$shift" for MT in
                                                                                                              (
            Matrix, Diagonal, Bidiagonal, Tridiagonal
        ),
        Tset in (Tsingle, Tdouble),
        Bdims2 in ((), (4,)),
        Tt in Tset,
        TA in Tset,
        TB in Tset,
        tscale in Tt.((0.1, 1, 10)),
        shift in (true, false)

        t = tscale * randn(Tt)
        if MT <: Bidiagonal
            A = MT(randn(TA, n, n), :U)
        else
            A = MT(randn(TA, n, n))
        end
        B = randn(TB, n, Bdims2...)
        T = Base.promote_eltype(t, A, B)
        rT = real(T)
        @inferred expv(t, A, B; shift=shift)
        @test expv(t, A, B; shift=shift) ≈ exp(Matrix(t * A)) * B
        @test eltype(expv(t, A, B; shift=shift)) === T
    end

    @testset "expv(t::$Tt, A::SparseMatrixCSC{$TA}, B::Array{$TB,$(length(Bdims2)+1)}), tscale=$tscale, shift=$shift" for Tset in
                                                                                                                          (
            Tsingle, Tdouble
        ),
        Bdims2 in ((), (4,)),
        Tt in Tset,
        TA in Tset,
        TB in Tset,
        tscale in Tt.((0.1, 1, 10)),
        shift in (true, false)

        t = tscale * randn(Tt)
        A = VERSION ≥ v"1.1" ? sprandn(TA, n, n, 1 / n) : TA.(sprandn(n, n, 1 / n))
        B = randn(TB, n, Bdims2...)
        T = Base.promote_eltype(t, A, B)
        rT = real(T)
        @inferred expv(t, A, B; shift=shift)
        @test expv(t, A, B; shift=shift) ≈ exp(Matrix(t * A)) * B
        @test eltype(expv(t, A, B; shift=shift)) === T
    end

    @testset "errors if tolerance too low" begin
        t = randn()
        A = randn(10, 10)
        B = randn(10, 10)

        @test_throws DomainError expv(t, A, B; tol=eps(Float64) / 100)
        @test_throws DomainError expv(big(t), big.(A), big.(B))
    end

    @testset "no errors for high norm" begin
        # https://github.com/sethaxen/ExponentialAction.jl/issues/10
        t = 20.0

        #! format: off
        A = [-4.19   0.0     8.75   0.0   0.0
              0.0  -57.45   33.26   0.0   0.0
              0.0    0.0  -175.05   0.0   0.0
              4.19  57.45    0.0  -87.53  6.28
              0.0    0.0    13.13   0.0  -6.28]
        #! format: on

        B = ones(size(A, 2))
        @test expv(t, A, B) ≈ exp(t * A) * B
    end
end
