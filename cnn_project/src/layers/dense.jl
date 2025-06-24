mutable struct Dense
    W::Matrix{Float32}
    b::Vector{Float32}
    σ::Function
    weight_decay::Float32
    last_input::Matrix{Float32}
    last_output::Matrix{Float32}
    last_pre_activation::Matrix{Float32}
    grad_pre_activation::Matrix{Float32}
end

function Dense(in_dim::Int, out_dim::Int, σ::Function; weight_decay=Float32(0.0001))
    xavier = sqrt(6 / (in_dim + out_dim))
    W = rand(Uniform(-xavier, xavier), out_dim, in_dim)
    b = zeros(Float32, out_dim)

    Dense(W, b, σ, weight_decay,
        Matrix{Float32}(undef, 0, 0),
        Matrix{Float32}(undef, 0, 0),
        Matrix{Float32}(undef, 0, 0),
        Matrix{Float32}(undef, 0, 0))
end

function (d::Dense)(x::AbstractArray)
    xmat = x isa Matrix && eltype(x) <: Float32 ? x : Float32.(reshape(x, :, size(x, ndims(x))))
    d.last_input = xmat
    d.last_pre_activation = d.W * xmat .+ d.b
    d.last_output = d.σ.(d.last_pre_activation)
    return d.last_output
end

function backward!(d::Dense, grad_output::Matrix{Float32})
    if size(grad_output, 2) == 1
        grad_output = reshape(grad_output, :, size(d.last_output, 2))
    end

    if size(d.grad_pre_activation) != size(grad_output)
        d.grad_pre_activation = similar(grad_output)
    end

    if d.σ == AD.relu
        @inbounds for i in eachindex(d.grad_pre_activation)
            d.grad_pre_activation[i] = d.last_pre_activation[i] > 0 ? grad_output[i] : 0f0
        end
    elseif d.σ == AD.sigmoid
        @inbounds for i in eachindex(d.grad_pre_activation)
            s = d.last_output[i]
            d.grad_pre_activation[i] = grad_output[i] * s * (1f0 - s)
        end
    else
        error("Unsupported activation function for backward pass")
    end

    grad_W = d.grad_pre_activation * d.last_input' .+ d.weight_decay .* d.W
    grad_b = vec(sum(d.grad_pre_activation, dims=2))
    grad_input = d.W' * d.grad_pre_activation

    return grad_W, grad_b, grad_input
end
