mutable struct BatchNorm1D
    channels::Int
    running_mean::Vector{Float32}
    running_var::Vector{Float32}
    gamma::Vector{Float32}
    beta::Vector{Float32}
    momentum::Float32
    eps::Float32
    is_training::Bool
    last_input::Array{Float32, 3}
    last_output::Array{Float32, 3}
    batch_mean::Vector{Float32}
    batch_var::Vector{Float32}
    grad_input::Array{Float32, 3}
end

function BatchNorm1D(channels::Int; momentum=Float32(0.1), eps=Float32(1e-5))
    BatchNorm1D(
        channels,
        zeros(Float32, channels),
        ones(Float32, channels),
        ones(Float32, channels),
        zeros(Float32, channels),
        momentum,
        eps,
        true,
        Array{Float32}(undef, 0, 0, 0),
        Array{Float32}(undef, 0, 0, 0),
        zeros(Float32, channels),
        ones(Float32, channels),
        Array{Float32}(undef, 0, 0, 0)
    )
end

function (bn::BatchNorm1D)(x::Array{Float32, 3})
    bn.last_input = x
    L, C, N = size(x)

    if bn.is_training
        bn.batch_mean = vec(mean(x, dims=(1, 3)))
        bn.batch_var = vec(var(x, dims=(1, 3), corrected=false))

        bn.running_mean .= (1 - bn.momentum) .* bn.running_mean .+ bn.momentum .* bn.batch_mean
        bn.running_var .= (1 - bn.momentum) .* bn.running_var .+ bn.momentum .* bn.batch_var

        x_norm = (x .- reshape(bn.batch_mean, 1, :, 1)) ./ sqrt.(reshape(bn.batch_var, 1, :, 1) .+ bn.eps)
    else
        x_norm = (x .- reshape(bn.running_mean, 1, :, 1)) ./ sqrt.(reshape(bn.running_var, 1, :, 1) .+ bn.eps)
    end

    bn.last_output = reshape(bn.gamma, 1, :, 1) .* x_norm .+ reshape(bn.beta, 1, :, 1)
    return bn.last_output
end

function backward!(bn::BatchNorm1D, grad_output::Array{Float32, 3})
    if !bn.is_training
        return grad_output
    end

    x = bn.last_input
    L, C, N = size(x)

    if size(bn.grad_input) != size(x)
        bn.grad_input = zeros(Float32, size(x))
    end

    x_centered = x .- reshape(bn.batch_mean, 1, :, 1)
    std_inv = 1.0 ./ sqrt.(bn.batch_var .+ bn.eps)

    grad_gamma = vec(sum(grad_output .* (x_centered .* reshape(std_inv, 1, :, 1)), dims=(1, 3)))
    grad_beta = vec(sum(grad_output, dims=(1, 3)))

    bn.grad_input .= grad_output .* reshape(bn.gamma .* std_inv, 1, :, 1)

    return grad_gamma, grad_beta, bn.grad_input
end
