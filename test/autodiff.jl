using ExponentialAction
using FiniteDifferences
using ForwardDiff
using ReverseDiff
using Zygote
using Test
using AbstractDifferentiation: AD

expv_explicit(t, A, B) = exp(t * A) * B

expv_sequence_explicit(ts, A, B) = map(t -> expv_explicit(t, A, B), ts)

function expv_jacobians(ba, t, A, B; f=expv, kwargs...)
    n = size(A, 2)
    tjac = only(AD.jacobian(ba, tvec -> f(tvec[1], A, B; kwargs...), [t]))
    Ajac = only(AD.jacobian(ba, Avec -> f(t, reshape(Avec, n, n), B; kwargs...), vec(A)))
    Bjac = only(AD.jacobian(ba, B -> f(t, A, B; kwargs...), B))
    return tjac, Ajac, Bjac
end

function expv_sequence_jacobians(ba, ts, A, B; f=expv_sequence, kwargs...)
    n = size(A, 2)
    tsjac = only(AD.jacobian(ba, ts -> reduce(vcat, f(ts, A, B; kwargs...)), collect(ts)))
    Ajac = only(
        AD.jacobian(
            ba, Avec -> reduce(vcat, f(ts, reshape(Avec, n, n), B; kwargs...)), vec(A)
        ),
    )
    Bjac = only(AD.jacobian(ba, B -> reduce(vcat, f(ts, A, B; kwargs...)), B))
    return tsjac, Ajac, Bjac
end

function expv_sequence_range_jacobians(ba, ts, A, B; f=expv_sequence, kwargs...)
    n = size(A, 2)
    tmin = ts[begin]
    tmax = ts[end]
    npoints = length(ts)
    tmin_jac = only(
        AD.jacobian(
            ba,
            tmin -> reduce(vcat, f(range(tmin[1], tmax, npoints), A, B; kwargs...)),
            [tmin],
        ),
    )
    tmax_jac = only(
        AD.jacobian(
            ba,
            tmax -> reduce(vcat, f(range(tmin, tmax[1], npoints), A, B; kwargs...)),
            [tmax],
        ),
    )
    Ajac = only(
        AD.jacobian(
            ba, Avec -> reduce(vcat, f(ts, reshape(Avec, n, n), B; kwargs...)), vec(A)
        ),
    )
    Bjac = only(AD.jacobian(ba, B -> reduce(vcat, f(ts, A, B; kwargs...)), B))
    return tmin_jac, tmax_jac, Ajac, Bjac
end

@testset "automatic differentiation" begin
    @testset "expv" begin
        t = rand()
        A = randn(10, 10)
        B = randn(10)
        fd_backend = AD.FiniteDifferencesBackend()
        backends = [
            "ForwardDiff" => AD.ForwardDiffBackend(),
            "ReverseDiff" => AD.ReverseDiffBackend(),
            "Zygote" => AD.ZygoteBackend(),
        ]
        tjac_exp, Ajac_exp, Bjac_exp = expv_jacobians(fd_backend, t, A, B; f=expv_explicit)
        @testset "$ba_name" for (ba_name, ba) in backends
            @testset for shift in (true, false)
                tjac, Ajac, Bjac = expv_jacobians(ba, t, A, B; shift)
                @test tjac ≈ tjac_exp atol = 1e-9 rtol = 1e-9
                @test Ajac ≈ Ajac_exp atol = 1e-9 rtol = 1e-9
                @test Bjac ≈ Bjac_exp atol = 1e-9 rtol = 1e-9
            end
        end
    end
    @testset "expv_sequence" begin
        tmin = 10 * rand()
        tmax = tmin + 1
        npoints = 10
        ts = range(tmin, tmax, npoints)
        A = randn(5, 5)
        B = randn(5)
        fd_backend = AD.FiniteDifferencesBackend()
        backends = [
            "ForwardDiff" => AD.ForwardDiffBackend(),
            "ReverseDiff" => AD.ReverseDiffBackend(),
            "Zygote" => AD.ZygoteBackend(),
        ]
        @testset "ts::Vector" begin
            tjac_exp, Ajac_exp, Bjac_exp = expv_sequence_jacobians(
                fd_backend, collect(ts), A, B; f=expv_sequence_explicit
            )
            @testset "$ba_name" for (ba_name, ba) in backends
                @testset for shift in (true, false)
                    tjac, Ajac, Bjac = expv_sequence_jacobians(ba, ts, A, B; shift)
                    @test tjac ≈ tjac_exp atol = 1e-9 rtol = 1e-9
                    @test Ajac ≈ Ajac_exp atol = 1e-9 rtol = 1e-9
                    @test Bjac ≈ Bjac_exp atol = 1e-9 rtol = 1e-9
                end
            end
        end
        @testset "ts::StepRangeLen" begin
            tmin_jac_exp, tmax_jac_exp, Ajac_exp, Bjac_exp = expv_sequence_range_jacobians(
                fd_backend, ts, A, B; f=expv_sequence_explicit
            )
            # Zygote currently can't differentiate through StepRangeLen
            # see https://github.com/FluxML/Zygote.jl/issues/550
            @testset "$ba_name" for (ba_name, ba) in
                                    filter(((k, v),) -> k !== "Zygote", backends)
                @testset for shift in (true, false)
                    tmin_jac, tmax_jac, Ajac, Bjac = expv_sequence_range_jacobians(
                        ba, ts, A, B; shift
                    )
                    @test tmin_jac ≈ tmin_jac_exp atol = 1e-9 rtol = 1e-9
                    @test tmax_jac ≈ tmax_jac_exp atol = 1e-9 rtol = 1e-9
                    @test Ajac ≈ Ajac_exp atol = 1e-9 rtol = 1e-9
                    @test Bjac ≈ Bjac_exp atol = 1e-9 rtol = 1e-9
                end
            end
        end
    end
end
