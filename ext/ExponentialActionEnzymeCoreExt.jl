module ExponentialActionEnzymeCoreExt

using ExponentialAction: ExponentialAction
using EnzymeCore: EnzymeRules

EnzymeRules.inactive(::typeof(ExponentialAction._parameters), t, A, ncols_B, degree_max, ℓ, tol) = nothing
EnzymeRules.inactive(::typeof(ExponentialAction._opnormInf), B) = nothing
EnzymeRules.inactive(::typeof(ExponentialAction.default_tol), args...) = nothing
EnzymeRules.inactive(::typeof(ExponentialAction.get_shift), A) = nothing

end  # module
