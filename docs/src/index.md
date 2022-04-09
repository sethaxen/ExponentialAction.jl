```@meta
CurrentModule = ExponentialAction
```

# ExponentialAction

## Introduction

ExponentialAction is a lightweight package that implements the action of the [Matrix exponential](https://en.wikipedia.org/wiki/Matrix_exponential) using the algorithm of Al-Mohy and Higham [^AlMohyHigham2011][^Expmv].
For details, see the docstring of [`expv`](@ref).

[^AlMohyHigham2011]: Al-Mohy, Awad H. and Higham, Nicholas J. (2011) Computing the Action of the Matrix Exponential, with an Application to Exponential Integrators.
    SIAM Journal on Scientific Computing, 33 (2). pp. 488-511. ISSN 1064-8275
    doi: [10.1137/100788860](https://doi.org/10.1137/100788860),
    eprint: [eprints.maths.manchester.ac.uk/id/eprint/1591](http://eprints.maths.manchester.ac.uk/id/eprint/1591)
[^Expmv]: [https://github.com/higham/expmv](https://github.com/higham/expmv)

This is the same algorithm used by SciPy's [`expm_multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html).

While `exp(X)` is only defined in LinearAlgebra for `Diagonal`, `Symmetric{<:Real}`/`Hermitian`, and `StridedMatrix`, `expv` can take an arbitrary matrix type.

## Example

```@repl 1
using LinearAlgebra, ExponentialAction
A = [1.0 2.0; -2.0 3.0];
B = [1.0, 2.0];
exp(A) * B
expv(1, A, B)
exp(2A) * B
expv(2, A, B)
```

## Automatic Differentiation (AD)

Special care has been taken to ensure that `expv` can be differentiated using [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl), [`ReverseDiff.jl`](https://github.com/JuliaDiff/ReverseDiff.jl), [`ChainRules.jl`](https://github.com/JuliaDiff/ChainRules.jl/)-compatible packages (e.g. [`Zygote.jl`](https://github.com/FluxML/Zygote.jl)), and likely others as a result.

This has been achieved by avoiding type constraints, not mutating any arrays, and marking operations only used for control flow as being non-differentiable.

## Related Packages

[`ExponentialUtilities.jl`](https://github.com/SciML/ExponentialUtilities.jl) and [`Expokit.jl`](https://github.com/acroy/Expokit.jl) both implement an approximation to the action of the matrix exponential using Krylov subspaces.
Which package is most efficient or useful depends on the choice of matrix and whether derivatives are needed.
If efficiency is important, we recommend choosing from the packages by benchmarking against several of your matrices (if applicable).

!!! note
    The packages use different default tolerances.
    `ExponentialAction.jl` is more strict.
    For a fair comparison, select similar tolerances.

