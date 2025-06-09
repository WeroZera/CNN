module NN

using ..AD
using Random
using LinearAlgebra
using Statistics
using Printf

export Dense, Chain, relu, sigmoid, softmax, binarycrossentropy, params, update!, DataLoader, Adam, Dropout, clip_gradients!, train_model

const ADValue = AD.ADValue

# Define activation functions first
relu(x) = x > 0 ? x : Float32(0)
sigmoid(x) = Float32(1) / (Float32(1) + exp(-x))
softmax(xs) = begin
    exps = exp.(xs .- maximum(xs))
    exps ./ sum(exps)
end

# Define Adam optimizer before it's used
mutable struct Adam
    lr::Float32
    β1::Float32
    β2::Float32
    ϵ::Float32
    state::Dict{Any, Tuple{Array{Float32}, Array{Float32}, Int}}
end

function Adam(lr=Float32(0.001), β1=Float32(0.9), β2=Float32(0.999), ϵ=Float32(1e-8))
    Adam(lr, β1, β2, ϵ, Dict{Any, Tuple{Array{Float32}, Array{Float32}, Int}}())
end

# Add Dropout layer
mutable struct Dropout
    p::Float32
    is_training::Bool
end

function Dropout(p=Float32(0.5))
    Dropout(p, true)
end

function (d::Dropout)(x::AbstractArray)
    if !d.is_training
        return x
    end
    mask = rand(Float32, size(x)) .> d.p
    return x .* mask ./ (Float32(1) - d.p)
end

function backward!(d::Dropout, grad_output::Matrix{Float32})
    if !d.is_training
        return grad_output
    end
    mask = rand(Float32, size(grad_output)) .> d.p
    return grad_output .* mask ./ (Float32(1) - d.p)
end

# Then define Dense layer
mutable struct Dense
    W::Matrix{Float32}
    b::Vector{Float32}
    σ::Function
    weight_decay::Float32  # Add weight decay parameter
    # Cache for backward pass
    last_input::Matrix{Float32}
    last_output::Matrix{Float32}
    last_pre_activation::Matrix{Float32}
end

function Dense(in_dim::Int, out_dim::Int, σ::Function; weight_decay=Float32(0.0001))
    # Initialize with He initialization
    W = Float32.(randn(out_dim, in_dim) .* sqrt(2 / in_dim))
    b = zeros(Float32, out_dim)

    # Initialize cache
    last_input = Matrix{Float32}(undef, 0, 0)
    last_output = Matrix{Float32}(undef, 0, 0)
    last_pre_activation = Matrix{Float32}(undef, 0, 0)

    Dense(W, b, σ, weight_decay, last_input, last_output, last_pre_activation)
end

function (d::Dense)(x::AbstractArray)
    # Convert input to Float32 matrix
    xmat = x isa Matrix ? x : reshape(x, :, 1)
    if !(eltype(xmat) <: Float32)
        xmat = Float32.(xmat)
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

    # Compute gradient of activation
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

    # Compute gradients with L2 regularization
    grad_W = grad_pre_activation * d.last_input' .+ d.weight_decay .* d.W
    grad_b = vec(sum(grad_pre_activation, dims=2))
    grad_input = d.W' * grad_pre_activation

    return grad_W, grad_b, grad_input
end

# Define Chain
mutable struct Chain
    layers::Vector{Union{Dense,Dropout,Chain}}  # Add Dropout to supported layers
    grad_cache::Vector{Any}
end

# Internal constructor
function Chain(layers::Vector{Union{Dense,Dropout,Chain}}, grad_cache::Vector{Any})
    new(layers, grad_cache)
end

# External constructor for varargs
function Chain(layers::Union{Dense,Dropout,Chain}...)
    layers_vec = collect(layers)
    grad_cache = [nothing for _ in layers_vec]
    Chain(layers_vec, grad_cache)
end

function (c::Chain)(x)
    for l in c.layers
        x = l(x)
    end
    x
end

# Backward pass through the chain
function backward!(c::Chain, grad_output::Matrix{Float32})
    # Ensure grad_output has the right shape for the last layer
    if size(grad_output, 2) == 1
        grad_output = reshape(grad_output, :, 1)
    end

    # Backward through layers in reverse order
    for i in length(c.layers):-1:1
        layer = c.layers[i]
        if layer isa Dense
            grad_W, grad_b, grad_output = backward!(layer, grad_output)
            c.grad_cache[i] = (grad_W, grad_b)
        elseif layer isa Dropout
            grad_output = backward!(layer, grad_output)
            c.grad_cache[i] = nothing
        else
            error("Unsupported layer type for backward pass")
        end
    end
    return c.grad_cache
end

# Update parameters using computed gradients
function update!(model::Chain, grads_cache, opt::Adam)
    for (i, layer) in enumerate(model.layers)
        if layer isa Dense && grads_cache[i] !== nothing
            grad_W, grad_b = grads_cache[i]

            for (param, grad) in zip((layer.W, layer.b), (grad_W, grad_b))
                m, v, t = get!(opt.state, param, (zeros(Float32, size(param)), zeros(Float32, size(param)), 0))
                t += 1
                m .= opt.β1 .* m .+ (Float32(1.0) - opt.β1) .* grad
                v .= opt.β2 .* v .+ (Float32(1.0) - opt.β2) .* (grad .^ 2)
                m̂ = m ./ (Float32(1.0) - opt.β1^t)
                ṽ = v ./ (Float32(1.0) - opt.β2^t)
                param .-= opt.lr .* m̂ ./ (sqrt.(ṽ) .+ opt.ϵ)
                opt.state[param] = (m, v, t)
            end
        end
    end
end

# Modify params to work with new Dense layer
function params(model)
    ps = Float32[]
    if model isa Dense
        append!(ps, vec(model.W))
        append!(ps, vec(model.b))
    elseif model isa Chain
        for layer in model.layers
            append!(ps, params(layer))
        end
    end
    return ps
end

binarycrossentropy(ŷ, y) = AD.binarycrossentropy(ŷ, y)

# DataLoader definition
struct DataLoader
    data::Tuple{Array{Float32,2}, Array{Float32,1}}
    batchsize::Int
    shuffle::Bool
end

function DataLoader(data::Tuple{Array{Float32,2},Array{Float32,1}}; batchsize::Int=64, shuffle::Bool=true)
    X, y = data
    # Ensure X is in the right shape (features × samples)
    if size(X, 1) < size(X, 2)
        X = transpose(X)
    end
    if shuffle
        idx = shuffle!(collect(1:size(X, 2)))
        return DataLoader((X[:, idx], y[idx]), batchsize, shuffle)
    end
    return DataLoader(data, batchsize, shuffle)
end

function Base.iterate(dl::DataLoader, state=1)
    if state > size(dl.data[1], 2)
        return nothing
    end
    end_idx = min(state + dl.batchsize - 1, size(dl.data[1], 2))
    # Ensure we return data in the correct shape
    batch = (dl.data[1][:, state:end_idx],
             dl.data[2][state:end_idx])
    return batch, state + dl.batchsize
end

function clip_gradients!(grads_cache, threshold::Float32)
    for i in 1:length(grads_cache)
        if grads_cache[i] !== nothing
            grad_W, grad_b = grads_cache[i]
            norm = sqrt(sum(grad_W.^2) + sum(grad_b.^2))
            if norm > threshold
                scaling = threshold / norm
                grad_W .*= scaling
                grad_b .*= scaling
                grads_cache[i] = (grad_W, grad_b)
            end
        end
    end
end


function loss(model, x, y)
    y_pred = vec(model(x))
    ϵ = Float32(1e-7)
    ŷ = clamp.(y_pred, ϵ, 1f0 - ϵ)
    return -mean(y .* log.(ŷ) .+ (1f0 .- y) .* log.(1f0 .- ŷ))
end

function grad(model, x, y)
    y_pred = vec(model(x))
    grad_pred = (y_pred .- y) ./ (y_pred .* (1f0 .- y_pred))
    grad_pred ./= length(y)
    return reshape(grad_pred, :, 1)
end

function train_model(model, dataset, test_X, test_y, opt, epochs, patience)
    best_val_loss = Inf
    patience_counter = 0
    best_model_state = nothing
    current_model = deepcopy(model)

    for epoch in 1:epochs
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for layer in current_model.layers
            if layer isa Dropout
                layer.is_training = true
            end
        end

        epoch_time = @elapsed begin
            for (x, y) in dataset
                loss_val = loss(current_model, x, y)
                grad_output = grad(current_model, x, y)

                grads_cache = backward!(current_model, reshape(grad_output, :, 1))
                clip_gradients!(grads_cache, 5.0f0)
                update!(current_model, grads_cache, opt)

                total_loss += loss_val
                total_acc += mean((vec(current_model(x)) .> 0.5) .== (y .> 0.5))
                num_batches += 1
            end
        end

        for layer in current_model.layers
            if layer isa Dropout
                layer.is_training = false
            end
        end

        train_loss = total_loss / num_batches
        train_acc = total_acc / num_batches

        ŷ_test = vec(current_model(test_X))
        ϵ = Float32(1e-7)
        ŷ_clamped = clamp.(ŷ_test, ϵ, 1f0 - ϵ)
        test_loss = -mean(test_y .* log.(ŷ_clamped) .+ (1f0 .- test_y) .* log.(1f0 .- ŷ_clamped))
        test_acc = mean((ŷ_test .> 0.5) .== (test_y .> 0.5))

        @printf "Epoch: %-2d (%.2fs)   Train: (l: %.4f, a: %.4f)   Test: (l: %.4f, a: %.4f)\n" epoch epoch_time train_loss train_acc test_loss test_acc
    end

    return current_model
end

end # module
