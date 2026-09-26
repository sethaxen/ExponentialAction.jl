using ExponentialAction
using FiniteDifferences
using ForwardDiff
using ReverseDiff
using Zygote
using Test
using AbstractDifferentiation: AbstractDifferentiation as AD

function expv_jacobians(ba, t, A, B; f = expv, kwargs...)
    n = size(A, 2)
    tjac = only(AD.jacobian(ba, tvec -> f(tvec[1], A, B; kwargs...), [t]))
    Ajac = only(AD.jacobian(ba, Avec -> f(t, reshape(Avec, n, n), B; kwargs...), vec(A)))
    Bjac = only(AD.jacobian(ba, B -> f(t, A, B; kwargs...), B))
    return tjac, Ajac, Bjac
end

function expv_sequence_jacobians(ba, ts, A, B; f = expv_sequence, kwargs...)
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

function expv_sequence_range_jacobians(ba, ts, A, B; f = expv_sequence, kwargs...)
    n = size(A, 2)
    tmin = ts[begin]
    tmax = ts[end]
    npoints = length(ts)
    tmin_jac = only(
        AD.jacobian(
            ba,
            tmin -> reduce(vcat, f(range(tmin[1], tmax; length = npoints), A, B; kwargs...)),
            [tmin],
        ),
    )
    tmax_jac = only(
        AD.jacobian(
            ba,
            tmax -> reduce(vcat, f(range(tmin, tmax[1]; length = npoints), A, B; kwargs...)),
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
        tjac_exp, Ajac_exp, Bjac_exp = expv_jacobians(fd_backend, t, A, B; f = expv_explicit)
        @testset "$ba_name" for (ba_name, ba) in backends
            @testset for shift in (true, false)
                tjac, Ajac, Bjac = expv_jacobians(ba, t, A, B; shift)
                @test tjac ≈ tjac_exp atol = 1.0e-9 rtol = 1.0e-9
                @test Ajac ≈ Ajac_exp atol = 1.0e-9 rtol = 1.0e-9
                @test Bjac ≈ Bjac_exp atol = 1.0e-9 rtol = 1.0e-9
            end
        end
    end
    @testset "expv_sequence" begin
        tmin = 10 * rand()
        tmax = tmin + 1
        npoints = 10
        ts = range(tmin, tmax; length = npoints)
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
                fd_backend, collect(ts), A, B; f = expv_sequence_explicit
            )
            @testset "$ba_name" for (ba_name, ba) in backends
                @testset for shift in (true, false)
                    tjac, Ajac, Bjac = expv_sequence_jacobians(ba, ts, A, B; shift)
                    @test tjac ≈ tjac_exp atol = 1.0e-9 rtol = 1.0e-9
                    @test Ajac ≈ Ajac_exp atol = 1.0e-9 rtol = 1.0e-9
                    @test Bjac ≈ Bjac_exp atol = 1.0e-9 rtol = 1.0e-9
                end
            end
        end
        @testset "ts::StepRangeLen" begin
            tmin_jac_exp, tmax_jac_exp, Ajac_exp, Bjac_exp = expv_sequence_range_jacobians(
                fd_backend, ts, A, B; f = expv_sequence_explicit
            )
            # Zygote currently can't differentiate through StepRangeLen
            # see https://github.com/FluxML/Zygote.jl/issues/550
            @testset "$ba_name" for (ba_name, ba) in
                filter(((k, v),) -> k !== "Zygote", backends)
                @testset for shift in (true, false)
                    tmin_jac, tmax_jac, Ajac, Bjac = expv_sequence_range_jacobians(
                        ba, ts, A, B; shift
                    )
                    @test tmin_jac ≈ tmin_jac_exp atol = 1.0e-9 rtol = 1.0e-9
                    @test tmax_jac ≈ tmax_jac_exp atol = 1.0e-9 rtol = 1.0e-9
                    @test Ajac ≈ Ajac_exp atol = 1.0e-9 rtol = 1.0e-9
                    @test Bjac ≈ Bjac_exp atol = 1.0e-9 rtol = 1.0e-9
                end
            end
        end
    end
end

@testset "expv dual at zero-primal B (issue #59)" begin
    # The Taylor convergence check strips duals; with an exactly-zero-primal
    # B carrying nonzero partials, the primal check 0 ≤ tol·0 fired at j=1 and
    # the returned duals carried only the first-order term.
    t = 0.1
    A = [0.0 -0.5; 0.5 0.0]
    x = [1.0, 0.0]
    expected = exp(t * A) * x

    # Zero primal, seeded duals (e.g. a state trajectory initialized at zero)
    B = [ForwardDiff.Dual{ForwardDiff.Tag{Nothing,Float64}}(0.0, 1.0),
         ForwardDiff.Dual{ForwardDiff.Tag{Nothing,Float64}}(0.0, 0.0)]
    y = expv(t, A, B)
    @test ForwardDiff.value.(y) ≈ zeros(2) atol = 1.0e-12
    @test [ForwardDiff.partials(yi)[1] for yi in y] ≈ expected atol = 1.0e-12

    # value-preserving sanity: nonzero primal remains exact
    B2 = [ForwardDiff.Dual{ForwardDiff.Tag{Nothing,Float64}}(1.0, 1.0),
          ForwardDiff.Dual{ForwardDiff.Tag{Nothing,Float64}}(0.0, 0.0)]
    y2 = expv(t, A, B2)
    @test ForwardDiff.value.(y2) ≈ expected atol = 1.0e-12
    @test [ForwardDiff.partials(yi)[1] for yi in y2] ≈ expected atol = 1.0e-12
end
