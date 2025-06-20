mutable struct Conv1D
    kernel_size::Int
    in_channels::Int
    out_channels::Int
    weight::Array{Float32, 3}
    bias::Vector{Float32}
    activation::Function
    last_input::Array{Float32, 3}
    last_output::Array{Float32, 3}
end

function Conv1D(k::Int, c_in::Int, c_out::Int, activation=relu)
    # Use He initialization for ReLU activations
    if activation == AD.relu
        he_std = sqrt(2.0 / (k * c_in))
        weight = randn(Float32, k, c_in, c_out) .* he_std
    else
        # Xavier initialization for other activations
        xavier = sqrt(6.0 / (k * c_in + c_out))
        weight = rand(Uniform(-xavier, xavier), k, c_in, c_out)
    end

    bias = zeros(Float32, c_out)
    last_input = Array{Float32}(undef, 0, 0, 0)
    last_output = Array{Float32}(undef, 0, 0, 0)
    Conv1D(k, c_in, c_out, weight, bias, activation, last_input, last_output)
end

function (c::Conv1D)(x::Array{Float32, 3})
    c.last_input = x
    L, C, N = size(x)
    k = c.kernel_size
    out_len = L - k + 1
    y = Array{Float32}(undef, out_len, c.out_channels, N)

    # Pre-allocate col matrix to avoid repeated allocations
    col = Array{Float32}(undef, k * C, out_len)

    # Reshape weight once
    kernels = reshape(c.weight, k * C, c.out_channels)

    @inbounds for n in 1:N
        # Fill col matrix more efficiently
        for i in 1:out_len
            @views col[:, i] = reshape(x[i:i+k-1, :, n], :)
        end

        # Single matrix multiplication
        z = kernels' * col

        # Add bias and apply activation in one pass - avoid temporary allocations
        @views for i in 1:out_len, j in 1:c.out_channels
            y[i, j, n] = c.activation(z[j, i] + c.bias[j])
        end
    end
    c.last_output = y

    return y
end


function backward!(c::Conv1D, grad_output::Array{Float32,3})
    x = c.last_input
    L, C, N = size(x)
    k = c.kernel_size
    out_len = L - k + 1

    grad_weight = zeros(Float32, k, C, c.out_channels)
    grad_bias = zeros(Float32, c.out_channels)
    grad_input = zeros(Float32, L, C, N)

    # Pre-allocate for efficiency
    col = Array{Float32}(undef, k * C, out_len)
    kernels = reshape(c.weight, k * C, c.out_channels)

    @inbounds for n in 1:N
        # Fill col matrix
        for i in 1:out_len
            @views col[:, i] = reshape(x[i:i+k-1, :, n], :)
        end

        # Compute gradients more efficiently
        grad_out_n = view(grad_output, :, :, n)

        # Grad w.r.t. weights: grad_output * col^T
        grad_weight_n = grad_out_n' * col'  # (out_channels, k*C)
        for oc in 1:c.out_channels
            grad_weight[:, :, oc] .+= reshape(grad_weight_n[oc, :], k, C)
        end

        # Grad w.r.t. bias
        grad_bias .+= vec(sum(grad_out_n, dims=1))

        # Grad w.r.t. input: weight^T * grad_output
        grad_input_n = kernels * grad_out_n'  # (k*C, out_len)
        for i in 1:out_len
            @views grad_input[i:i+k-1, :, n] .+= reshape(grad_input_n[:, i], k, C)
        end
    end

    return grad_weight, grad_bias, grad_input
end