using DifferentiationInterface
using DifferentiationInterfaceTest
using Enzyme: Enzyme
using ExponentialAction
using ForwardDiff: ForwardDiff
using LinearAlgebra
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff
using StableRNGs
using Zygote: Zygote
using CSV

struct SumExpvFixedA{T}
    A::T
end
(f::SumExpvFixedA)(v) = sum(expv(one(eltype(v)), f.A, v))

struct SumExpvFixedV{F,T}
    A_constructor::F
    v::T
end
(f::SumExpvFixedV)(A_arg) = sum(expv(one(eltype(A_arg)), f.A_constructor(A_arg), f.v))

nrows_v = [10, 100, 1_000]

rng = StableRNG(42)

vs = [randn(rng, n) for n in nrows_v]
As = []
for n in nrows_v
    A = randn(rng, n, n)
    push!(As, A)
    push!(As, Diagonal(A))
end

scenarios = Scenario[]
for v in vs, A in As
    size(v, 1) == size(A, 1) || continue
    if size(A, 1) > 10 && A isa Matrix
        continue
    end
    name_base = "A::$(nameof(typeof(A)))-$(size(A, 1))"
    push!(scenarios, Scenario{:gradient,:out}(SumExpvFixedA(A), v; name="v:$name_base"))
    A_constructor, A_arg = if A isa Matrix
        identity, A
    elseif A isa Diagonal
        Diagonal, diag(A)
    end
    push!(
        scenarios,
        Scenario{:gradient,:out}(
            SumExpvFixedV(A_constructor, v), A_arg; name="A:$name_base"
        ),
    )
end

backends = [
    AutoForwardDiff(),
    AutoReverseDiff(),
    AutoZygote(),
    AutoEnzyme(; function_annotation=Enzyme.Const),
    AutoMooncake(),
]

results = benchmark_differentiation(backends, scenarios; logging=true)
CSV.write("benchmark_ad.csv", results)
display(results)
