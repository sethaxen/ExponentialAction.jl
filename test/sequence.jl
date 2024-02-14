using ExponentialAction, LinearAlgebra, Test

Tdouble = (Float64, ComplexF64)

@testset "expv_sequence" begin
    n = 5
    ts_base = 1:0.1:2
    @testset "expv_sequence(ts, A::$Matrix{$TA}, B::Array{$TB,$(length(Bdims2)+1)}), tscale=$tscale, shift=$shift" for Tset in
                                                                                                                       (
            Tdouble,
        ),
        Bdims2 in ((), (4,)),
        TA in Tset,
        TB in Tset,
        tscale in (0.1, 1, 50),
        shift in (true, false)

        ts = tscale * ts_base
        A = randn(TA, n, n)
        B = randn(TB, n, Bdims2...)
        T = float(Base.promote_eltype(ts, A, B))
        rT = real(T)
        Fs_exp = expv_sequence_explicit(ts, A, B)
        Fs_range = @inferred expv_sequence(ts, A, B; shift)
        @test Fs_range ≈ Fs_exp
        @test eltype(first(Fs_range)) === T
        ts_vec = collect(ts)
        Fs_vec = @inferred expv_sequence(ts_vec, A, B; shift)
        @test Fs_vec ≈ Fs_exp
        @test eltype(first(Fs_vec)) === T
    end

    @testset "length(ts) = 1" begin
        t = rand()
        A = randn(5, 5)
        B = randn(5, 2)
        @test expv_sequence([t], A, B) ≈ [expv(t, A, B)]
        @test expv_sequence(t:1:t, A, B) ≈ [expv(t, A, B)]
    end

    @testset "ts all identical" begin
        A = randn(5, 5)
        B = randn(5, 2)
        Fs = expv_sequence(ones(5), A, B)
        @test all(Fs .≈ Ref(expv(1, A, B)))
        Fs = expv_sequence(zeros(5), A, B)
        @test all(Fs .≈ Ref(B))
    end
end
