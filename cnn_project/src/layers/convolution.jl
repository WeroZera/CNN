mutable struct Conv1D
    kernel_size::Int
    in_channels::Int
    out_channels::Int
    weight::Array{Float32, 3}
    bias::Vector{Float32}
    activation::Function
    last_input::Array{Float32, 3}
    last_output::Array{Float32, 3}
    col::Array{Float32, 2}
    kernels::Array{Float32, 2}
end

function Conv1D(k::Int, c_in::Int, c_out::Int, activation=relu)
    if activation == AD.relu
        he_std = sqrt(2.0 / (k * c_in))
        weight = randn(Float32, k, c_in, c_out) .* he_std
    else
        xavier = sqrt(6.0 / (k * c_in + c_out))
        weight = rand(Uniform(-xavier, xavier), k, c_in, c_out)
    end

    bias = zeros(Float32, c_out)

    Conv1D(
        k, c_in, c_out, weight, bias, activation,
        Array{Float32}(undef, 0, 0, 0),  # last_input
        Array{Float32}(undef, 0, 0, 0),  # last_output
        Array{Float32}(undef, 0, 0),     # col
        Array{Float32}(undef, 0, 0)      # kernels
    )
end

function (c::Conv1D)(x::Array{Float32, 3})
    c.last_input = x
    L, C, N = size(x)
    k = c.kernel_size
    out_len = L - k + 1
    y = Array{Float32}(undef, out_len, c.out_channels, N)

    c.col = Array{Float32}(undef, k * C, out_len)
    c.kernels = reshape(c.weight, k * C, c.out_channels)

    @inbounds for n in 1:N
        for i in 1:out_len
            @views c.col[:, i] = reshape(x[i:i+k-1, :, n], :)
        end

        z = c.kernels' * c.col

        @views for i in 1:out_len, j in 1:c.out_channels
            y[i, j, n] = c.activation(z[j, i] + c.bias[j])
        end
    end

    c.last_output = y
    return y
end

function backward!(c::Conv1D, grad_output::Array{Float32, 3})
    x = c.last_input
    L, C, N = size(x)
    k = c.kernel_size
    out_len = L - k + 1

    grad_weight = zeros(Float32, k, C, c.out_channels)
    grad_bias = zeros(Float32, c.out_channels)
    grad_input = zeros(Float32, L, C, N)

    for n in 1:N
        for i in 1:out_len
            @views c.col[:, i] = reshape(x[i:i+k-1, :, n], :)
        end

        grad_out_n = view(grad_output, :, :, n)

        grad_weight_n = grad_out_n' * c.col'
        for oc in 1:c.out_channels
            grad_weight[:, :, oc] .+= reshape(grad_weight_n[oc, :], k, C)
        end

        grad_bias .+= vec(sum(grad_out_n, dims=1))

        grad_input_n = c.kernels * grad_out_n'
        for i in 1:out_len
            @views grad_input[i:i+k-1, :, n] .+= reshape(grad_input_n[:, i], k, C)
        end
    end

    return grad_weight, grad_bias, grad_input
end
