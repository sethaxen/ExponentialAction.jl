using ExponentialAction

Tsingle = (Float32, ComplexF32)
Tdouble = (Float64, ComplexF64)

@testset "expv" begin
    n = 10
    @testset "expv(t::$Tt, A::$MT{$TA}, B::Array{$TB,$(length(Bdims2)+1)}), tscale=$tscale" for
        MT in (Matrix, Diagonal),
        Tset in (Tsingle, Tdouble),
        Bdims2 in ((), (4,)),
        Tt in Tset,
        TA in Tset,
        TB in Tset,
        tscale in Tt.((0.1, 1, 10)),
        shift in (true, false)

        t = tscale * randn(Tt)
        A = MT(randn(TA, n, n))
        B = randn(TB, n, Bdims2...)
        T = Base.promote_eltype(t, A, B)
        rT = real(T)
        @inferred expv(t, A, B; shift=shift)
        @test expv(t, A, B; shift=shift) ≈ exp(t * A) * B
        @test expv(t, A, B; shift=shift) ≈ exp(t * A) * B
        @test eltype(expv(t, A, B; shift=shift)) === T
    end
end
