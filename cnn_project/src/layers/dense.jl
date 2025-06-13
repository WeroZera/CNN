# Then define Dense layer
mutable struct Dense
    W::Matrix{Float32}
    b::Vector{Float32}
    σ::Function
    weight_decay::Float32    
    last_input::Matrix{Float32}
    last_output::Matrix{Float32}
    last_pre_activation::Matrix{Float32}
end

function Dense(in_dim::Int, out_dim::Int, σ::Function; weight_decay=Float32(0.0001))
    # Initialize with He initialization
    xavier = sqrt(6 / (in_dim + out_dim))
    W = rand(Uniform(-xavier, xavier), out_dim, in_dim)
    b = zeros(Float32, out_dim)

    # Initialize cache
    last_input = Matrix{Float32}(undef, 0, 0)
    last_output = Matrix{Float32}(undef, 0, 0)
    last_pre_activation = Matrix{Float32}(undef, 0, 0)

    Dense(W, b, σ, weight_decay, last_input, last_output, last_pre_activation)
end

function (d::Dense)(x::AbstractArray)
    # Convert input to Float32 matrix more efficiently
    if x isa Matrix && eltype(x) <: Float32
        xmat = x
    else
        xmat = Float32.(reshape(x, :, size(x, ndims(x))))
    end

    # Store input for backward pass
    d.last_input = xmat

    # Compute pre-activation
    d.last_pre_activation = d.W * xmat .+ d.b

    # Apply activation
    d.last_output = d.σ.(d.last_pre_activation)

    return d.last_output
end

# Direct gradient computation for backward pass
function backward!(d::Dense, grad_output::Matrix{Float32})
    # Ensure grad_output has the right shape
    if size(grad_output, 2) == 1
        grad_output = reshape(grad_output, :, size(d.last_output, 2))
    end

    # Compute gradient of activation more efficiently
    grad_pre_activation = similar(grad_output)
    if d.σ == relu
        @inbounds for i in eachindex(grad_pre_activation)
            grad_pre_activation[i] = d.last_pre_activation[i] > 0 ? grad_output[i] : Float32(0)
        end
    elseif d.σ == sigmoid
        @inbounds for i in eachindex(grad_pre_activation)
            s = d.last_output[i]
            grad_pre_activation[i] = grad_output[i] * s * (Float32(1) - s)
        end
    else
        error("Unsupported activation function for backward pass")
    end

    # Compute gradients with L2 regularization using BLAS operations
    grad_W = grad_pre_activation * d.last_input' .+ d.weight_decay .* d.W
    grad_b = vec(sum(grad_pre_activation, dims=2))
    grad_input = d.W' * grad_pre_activation

    return grad_W, grad_b, grad_input
end