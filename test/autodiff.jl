using ExponentialAction
using FiniteDifferences
using ForwardDiff
using ReverseDiff
using Zygote
using Test
using AbstractDifferentiation: AD

# function _expv_sequence_range(t_min, t_max, n, A, B; kwargs...)
# end

function expv_jacobians(ba, t, A, B; kwargs...)
    n = size(A, 2)
    tjac = AD.jacobian(fd_backend, tvec -> expv(tvec[1], A, B; kwargs...), [t])
    Ajac = AD.jacobian(fd_backend, Avec -> expv(t, reshape(Avec, n, n), B; kwargs...), vec(A))
    Bjac = AD.jacobian(fd_backend, B -> expv(t, A, B; kwargs...), B)
    return only.((tjac, Ajac, Bjac))
end

@testset "automatic differentiation" begin
    t = rand()
    A = randn(10, 10)
    B = randn(10)
    fd_backend = AD.FiniteDifferencesBackend()
    backends = [
        "ForwardDiff" => AD.ForwardDiffBackend(),
        "ReverseDiff" => AD.ReverseDiffBackend(),
        "Zygote" => AD.ZygoteBackend(),
    ]
    tjac_exp, Ajac_exp, Bjac_exp = expv_jacobians(fd_backend, t, A, B)

    @testset "$ba_name" for (ba_name, ba) in backends, shift in (true, false)
        tjac, Ajac, Bjac = expv_jacobians(ba, t, A, B; shift)
        @test tjac ≈ tjac_exp
        @test Ajac ≈ Ajac_exp
        @test Bjac ≈ Bjac_exp
    end
end
