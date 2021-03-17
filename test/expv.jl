using ExponentialAction, LinearAlgebra, SparseArrays, Test

Tsingle = (Float32, ComplexF32)
Tdouble = (Float64, ComplexF64)

@testset "expv" begin
    n = 10
    @testset "expv(t::$Tt, A::$MT{$TA}, B::Array{$TB,$(length(Bdims2)+1)}), tscale=$tscale, shift=$shift" for
        MT in (Matrix, Diagonal, Bidiagonal, Tridiagonal),
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

    @testset "expv(t::$Tt, A::SparseMatrixCSC{$TA}, B::Array{$TB,$(length(Bdims2)+1)}), tscale=$tscale, shift=$shift" for
        Tset in (Tsingle, Tdouble),
        Bdims2 in ((), (4,)),
        Tt in Tset,
        TA in Tset,
        TB in Tset,
        tscale in Tt.((0.1, 1, 10)),
        shift in (true, false)

        t = tscale * randn(Tt)
        A = sprandn(TA, n, n, 1 / n)
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
end
