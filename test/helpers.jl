using LinearAlgebra

expv_explicit(t, A, B) = exp(t * A) * B

expv_sequence_explicit(ts, A, B) = map(t -> expv_explicit(t, A, B), ts)

# linear operator implementing the minimal interface
struct LinOp{T}
    A::T
end
Base.size(op::LinOp, i) = size(op.A, i)
Base.eltype(op::LinOp) = eltype(op.A)
function LinearAlgebra.mul!(y, op::LinOp, x)
    LinearAlgebra.mul!(y, op.A, x)
    return y
end
Base.adjoint(op::LinOp) = LinOp(adjoint(op.A))


struct MatVecCountingLinOp{T}
    A::T
    matvecs::Ref{Int}
end
MatVecCountingLinOp(A) = MatVecCountingLinOp(A, Ref(0))
Base.size(op::MatVecCountingLinOp, i) = size(op.A, i)
Base.eltype(op::MatVecCountingLinOp) = eltype(op.A)
function LinearAlgebra.mul!(y, op::MatVecCountingLinOp, x)
    LinearAlgebra.mul!(y, op.A, x)
    op.matvecs[] += size(x, 2)
    return y
end
Base.adjoint(op::MatVecCountingLinOp) = MatVecCountingLinOp(adjoint(op.A), op.matvecs)
