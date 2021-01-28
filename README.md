# ExponentialAction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sethaxen.github.io/ExponentialAction.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sethaxen.github.io/ExponentialAction.jl/dev)
[![Build Status](https://github.com/sethaxen/ExponentialAction.jl/workflows/CI/badge.svg)](https://github.com/sethaxen/ExponentialAction.jl/actions)
[![Coverage](https://codecov.io/gh/sethaxen/ExponentialAction.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sethaxen/ExponentialAction.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

ExponentialAction is a lightweight, zero-dependency package that implements the action of the [Matrix exponential](https://en.wikipedia.org/wiki/Matrix_exponential) using the algorithm of Al-Mohy and Higham<sup>[1](#references)</sup>.

The API of this package is a single function `expv(t, A, B)` that computes `exp(t * A) * B` for a scalar `t`, matrix `A`, and matrix of vector `B`, without computing `exp(t * A)`.
For large matrices `A`, this is significantly less expensive than calling `exp(t * A) * B` directly.
For more information, see the docstring of `expv`.

## Related Packages

[ExponentialUtilities.jl](https://github.com/SciML/ExponentialUtilities.jl) and [Expokit.jl](https://github.com/acroy/Expokit.jl) both implement an approximation to the action of the matrix exponential using Krylov subspaces.

## References

1. Al-Mohy, Awad H. and Higham, Nicholas J. (2011) Computing the Action of the Matrix Exponential, with an Application to Exponential Integrators. SIAM Journal on Scientific Computing, 33 (2). pp. 488-511. ISSN 1064-8275
   doi: [10.1137/100788860](https://doi.org/10.1137/100788860),
   eprint: [eprints.maths.manchester.ac.uk/id/eprint/1591](http://eprints.maths.manchester.ac.uk/id/eprint/1591),
2. https://github.com/higham/expmv
