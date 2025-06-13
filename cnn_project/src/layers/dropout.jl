mutable struct Dropout
    p::Float32
    is_training::Bool
    mask::AbstractArray{Bool}
end

function Dropout(p=Float32(0.2))
    Dropout(p, true, BitArray(undef, 0))
end

function (d::Dropout)(x::AbstractArray)
    if d.is_training
        T = eltype(x)  
        d.mask = rand(T, size(x)) .>= T(d.p)  # generate mask with correct type
        return x .* d.mask ./ (T(1.0) - T(d.p))  # scale to preserve expectation
    else
        return x
    end
end

function backward!(d::Dropout, grad_output::AbstractArray)
    if d.is_training
        return grad_output .* d.mask ./ (Float32(1.0) - d.p)
    else
        return grad_output
    end
end