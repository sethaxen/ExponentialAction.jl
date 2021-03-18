using ExponentialAction, LinearAlgebra, SparseArrays, Test

Thalf = (Float16, ComplexF16)
Tsingle = (Float32, ComplexF32)
Tdouble = (Float64, ComplexF64)
tscales(::Type{T}) where {T} = (log(maxintfloat(real(T))) / 3) ./ (100, 10, 1)

@testset "expv" begin
    n = 10
    @testset "expv(t::$Tt, A::$MT{$TA}, B::Array{$TB,$(length(Bdims2)+1)}), tscale=$tscale, shift=$shift" for
        MT in (Matrix, Diagonal, Bidiagonal, Tridiagonal),
        Tset in (Thalf, Tsingle, Tdouble),
        Bdims2 in ((), (4,)),
        Tt in Tset,
        TA in Tset,
        TB in Tset,
        tscale in tscales(Tt),
        shift in (true, false)

        t = tscale * randn(Tt)
        if MT <: Bidiagonal
            A = MT(randn(TA, n, n), :U)
        else
            A = MT(randn(TA, n, n))
        end
        B = randn(TB, n, Bdims2...)
        T = Base.promote_eltype(t, A, B)
        Te = Base.promote_type(T, Float32)
        @inferred expv(t, A, B; shift=shift)
        @test expv(t, A, B; shift=shift) ≈ T.(exp(Matrix{Te}(t * A)) * B)
        @test eltype(expv(t, A, B; shift=shift)) === T
    end

    @testset "expv(t::$Tt, A::SparseMatrixCSC{$TA}, B::Array{$TB,$(length(Bdims2)+1)}), tscale=$tscale, shift=$shift" for
        Tset in (Thalf, Tsingle, Tdouble),
        Bdims2 in ((), (4,)),
        Tt in Tset,
        TA in Tset,
        TB in Tset,
        tscale in tscales(Tt),
        shift in (true, false)

        t = tscale * randn(Tt)
        A = VERSION ≥ v"1.1" ? sprandn(TA, n, n, 1 / n) : TA.(sprandn(n, n, 1 / n))
        B = randn(TB, n, Bdims2...)
        T = Base.promote_eltype(t, A, B)
        Te = Base.promote_type(T, Float32)
        @inferred expv(t, A, B; shift=shift)
        @test expv(t, A, B; shift=shift) ≈ T.(exp(Matrix{Te}(t * A)) * B)
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
