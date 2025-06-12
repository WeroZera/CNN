mutable struct MaxPool1D
    pool_size::Int
    last_input::Array{Float32, 3}
    mask::BitArray{3}
end

function MaxPool1D(pool_size::Int)
    MaxPool1D(pool_size, Array{Float32}(undef, 0, 0, 0), BitArray(undef, 0, 0, 0))
end

function (p::MaxPool1D)(x::Array{Float32, 3})
    p.last_input = x
    L, C, N = size(x)
    stride = p.pool_size
    out_len = Int(floor(L / stride))
    result = Array{Float32, 3}(undef, out_len, C, N)
    mask = falses(L, C, N)

    @inbounds for n in 1:N, c in 1:C
        for i in 1:out_len
            start_idx = (i-1)*stride + 1
            end_idx = i*stride
            @views window = x[start_idx:end_idx, c, n]
            maxval = maximum(window)
            result[i, c, n] = maxval

            # Find first occurrence of max value
            for j in 1:stride
                if window[j] == maxval
                    mask[start_idx + j - 1, c, n] = true
                    break
                end
            end
        end
    end
    p.mask = mask
    return result
end

function backward!(p::MaxPool1D, grad_output::Array{Float32, 3})
    # grad_output: (out_len, C, N)
    # mask: (L, C, N)
    L, C, N = size(p.last_input)
    stride = p.pool_size
    out_len = Int(floor(L / stride))
    grad_input = zeros(Float32, L, C, N)
    for n in 1:N, c in 1:C
        for i in 1:out_len
            # Find the max location in the window
            for j in 1:stride
                idx = (i-1)*stride + j
                if p.mask[idx, c, n]
                    grad_input[idx, c, n] = grad_output[i, c, n]
                    break  # Only the max gets the gradient
                end
            end
        end
    end
    return grad_input
end