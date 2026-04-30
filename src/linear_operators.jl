## Linear operators used internally
struct ShiftedLinearOperator{T, S <: Number}
    op::T
    shift::S
end

Base.size(op::ShiftedLinearOperator, i) = size(op.op, i)
Base.eltype(op::ShiftedLinearOperator) = promote_type(eltype(op.op), typeof(op.shift))
function LinearAlgebra.mul!(y, op::ShiftedLinearOperator, x)
    LinearAlgebra.mul!(y, op.op, x)
    LinearAlgebra.axpy!(op.shift, x, y)
    return y
end
Base.transpose(op::ShiftedLinearOperator) = ShiftedLinearOperator(transpose(op.op), op.shift)
Base.adjoint(op::ShiftedLinearOperator) = ShiftedLinearOperator(adjoint(op.op), conj(op.shift))

struct PowerLinearOperator{T}
    op::T
    pow::Int
end

Base.size(op::PowerLinearOperator, i) = size(op.op, i)
Base.eltype(op::PowerLinearOperator) = eltype(op.op)
function LinearAlgebra.mul!(y, op::PowerLinearOperator, x)
    copyto!(y, x)
    z = similar(y)
    for _ in 1:op.pow
        LinearAlgebra.mul!(z, op.op, y)
        copyto!(y, z)
    end
    return y
end
Base.transpose(op::PowerLinearOperator) = PowerLinearOperator(transpose(op.op), op.pow)
Base.adjoint(op::PowerLinearOperator) = PowerLinearOperator(adjoint(op.op), op.pow)
