module NN

using ..AD
using Random
using LinearAlgebra
using Statistics
using Printf
using Distributions

export Dense, Chain, relu, sigmoid, softmax, binarycrossentropy, params, update!,
       DataLoader, Adam, Dropout, clip_gradients!, train_model,
       Embedding, Conv1D, MaxPool1D, flatten, BatchNorm1D,
       set_eval_mode!, set_train_mode!

const ADValue = AD.ADValue

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

# Add Dropout layer
mutable struct Dropout
    p::Float32
    is_training::Bool
    mask::AbstractArray{Bool}
end

# Define Chain
mutable struct Chain
    layers::Vector{Any}
    grad_cache::Vector{Any}
    shape_cache::Vector{Any}  # Store shapes for backward pass
end

mutable struct Embedding
    vocab_size::Int
    embedding_dim::Int
    weight::Array{Float32, 2}
    last_indices::Vector{Int}
end

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

mutable struct MaxPool1D
    pool_size::Int
    last_input::Array{Float32, 3}
    mask::BitArray{3}
end

flatten(x) = reshape(x, :, size(x, 3))

# CNN Components


function Embedding(vocab_size::Int, embedding_dim::Int, weights::Array{<:Real,2})
    Embedding(vocab_size, embedding_dim, Float32.(weights), Int[])
end

function (e::Embedding)(x::AbstractArray{Int})
    inds = vec(x)  # flatten for indexing
    e.last_indices = inds
    emb = e.weight[:, inds]
    return reshape(emb, e.embedding_dim, size(x)...)
end

function Conv1D(k::Int, c_in::Int, c_out::Int, activation=relu)
    # Use He initialization for ReLU activations (better than Xavier for deep networks)
    if activation == relu
        he_std = sqrt(2.0 / (k * c_in))
        weight = randn(Float32, k, c_in, c_out) .* he_std
    else
        # Xavier/Glorot initialization for other activations
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

# Define activation functions first
relu(x) = x > 0 ? x : Float32(0)
relu(x::AbstractArray) = max.(x, Float32(0))

# LeakyReLU - often better than ReLU
leakyrelu(x, α=Float32(0.01)) = x > 0 ? x : α * x
leakyrelu(x::AbstractArray, α=Float32(0.01)) = max.(x, α .* x)

# ELU - exponential linear unit
elu(x, α=Float32(1.0)) = x > 0 ? x : α * (exp(x) - 1)
elu(x::AbstractArray, α=Float32(1.0)) = x .> 0 ? x : α .* (exp.(x) .- 1)

sigmoid(x) = Float32(1) / (Float32(1) + exp(-x))
sigmoid(x::AbstractArray) = Float32(1) ./ (Float32(1) .+ exp.(-x))

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

function Dropout(p=Float32(0.2))
    Dropout(p, true, BitArray(undef, 0))
end

function (d::Dropout)(x::AbstractArray)
    if d.is_training
        T = eltype(x)  # get the element type of x (e.g., Float32)
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


function backward!(e::Embedding, grad_output::Array{Float32, 2})
    grad_weight = zeros(Float32, size(e.weight))  # (embedding_dim, vocab_size)

    # More efficient gradient accumulation without dictionary
    for (i, idx) in enumerate(e.last_indices)
        grad_weight[:, idx] .+= grad_output[:, i]
    end

    # Average gradients for repeated indices more efficiently
    unique_indices = unique(e.last_indices)
    for idx in unique_indices
        # Count occurrences manually
        count = 0
        for j in e.last_indices
            if j == idx
                count += 1
            end
        end
        if count > 1
            grad_weight[:, idx] ./= count
        end
    end

    return grad_weight, unique_indices
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

# Internal constructor
function Chain(layers...)
    layer_list = collect(layers)
    grad_cache = [nothing for _ in layer_list]
    shape_cache = [nothing for _ in layer_list]
    Chain(layer_list, grad_cache, shape_cache)
end

# External constructor for varargs
function Chain(layers::Any...)
    layers_vec = collect(layers)
    grad_cache = [nothing for _ in layers_vec]
    shape_cache = [nothing for _ in layers_vec]
    Chain(layers_vec, grad_cache, shape_cache)
end

function (c::Chain)(x)
    for (i, l) in enumerate(c.layers)
        x = l(x)
        c.shape_cache[i] = size(x)
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
        elseif layer isa Conv1D
            # Reshape gradient to match the original Conv1D output
            if ndims(grad_output) == 2
                # Use the stored shape from forward pass
                if c.shape_cache[i] !== nothing
                    grad_output = reshape(grad_output, c.shape_cache[i])
                else
                    error("No shape stored for Conv1D layer")
                end
            end
            grad_W, grad_b, grad_output = backward!(layer, grad_output)
            c.grad_cache[i] = (grad_W, grad_b)
        elseif layer isa Embedding
            if ndims(grad_output) == 3
                embedding_dim = size(grad_output, 2)
                total_elements = size(grad_output, 1) * size(grad_output, 3)
                grad_output = reshape(grad_output, embedding_dim, total_elements)
            end
            grad_weight, used_indices = backward!(layer, grad_output)
            c.grad_cache[i] = (grad_weight, used_indices)
        elseif layer isa MaxPool1D
            # Call backward! for MaxPool1D to upsample the gradient
            grad_output = backward!(layer, grad_output)
            c.grad_cache[i] = nothing
        elseif layer isa BatchNorm1D
            # Reshape gradient to match the original BatchNorm1D output
            if ndims(grad_output) == 2
                # Use the stored shape from forward pass
                if c.shape_cache[i] !== nothing
                    grad_output = reshape(grad_output, c.shape_cache[i])
                else
                    error("No shape stored for BatchNorm1D layer")
                end
            end
            grad_gamma, grad_beta, grad_output = backward!(layer, grad_output)
            c.grad_cache[i] = (grad_gamma, grad_beta)
        elseif isa(layer, Function)
            if layer == flatten
                if i > 1 && c.shape_cache[i-1] !== nothing
                    grad_output = reshape(grad_output, c.shape_cache[i-1])
                else
                    error("No shape stored for flatten operation")
                end
                c.grad_cache[i] = nothing
            elseif layer == (x -> permutedims(x, (2, 1, 3)))
                # For permute, we need to reverse the permutation
                if ndims(grad_output) == 3
                    grad_output = permutedims(grad_output, (2, 1, 3))
                end
                c.grad_cache[i] = nothing
            else
                # For other function layers, just pass through
                c.grad_cache[i] = nothing
            end
        else
            error("Unsupported layer type for backward pass: $(typeof(layer))")
        end
    end
    return c.grad_cache
end

function update!(model::Chain, grads_cache, opt::Adam)
    for (i, layer) in enumerate(model.layers)
        grads = grads_cache[i]
        if grads === nothing
            continue
        end

        if layer isa Dense
            grad_W, grad_b = grads
            for (param, grad) in zip((layer.W, layer.b), (grad_W, grad_b))
                key = objectid(param)
                m, v, t = get!(opt.state, key, (zeros(Float32, size(param)), zeros(Float32, size(param)), 0))
                t += 1
                m .= opt.β1 .* m .+ (1 - opt.β1) .* grad
                v .= opt.β2 .* v .+ (1 - opt.β2) .* (grad .^ 2)
                m̂ = m ./ (1 - opt.β1^t + Float32(1e-8))
                ṽ = v ./ (1 - opt.β2^t + Float32(1e-8))
                # Learning rate scheduling: reduce by 10% every 5 epochs
                current_lr = opt.lr * Float32(0.9)^(div(t, 1000))  # Adjust divisor based on your batch size
                param .-= current_lr .* m̂ ./ (sqrt.(ṽ) .+ opt.ϵ)
                opt.state[key] = (m, v, t)
            end

        elseif layer isa Conv1D
            grad_W, grad_b = grads
            for (param, grad) in zip((layer.weight, layer.bias), (grad_W, grad_b))
                key = objectid(param)
                m, v, t = get!(opt.state, key, (zeros(Float32, size(param)), zeros(Float32, size(param)), 0))
                t += 1
                m .= opt.β1 .* m .+ (1 - opt.β1) .* grad
                v .= opt.β2 .* v .+ (1 - opt.β2) .* (grad .^ 2)
                m̂ = m ./ (1 - opt.β1^t + Float32(1e-8))
                ṽ = v ./ (1 - opt.β2^t + Float32(1e-8))
                # Learning rate scheduling
                current_lr = opt.lr * Float32(0.9)^(div(t, 1000))
                param .-= current_lr .* m̂ ./ (sqrt.(ṽ) .+ opt.ϵ)
                opt.state[key] = (m, v, t)
            end

        elseif layer isa Embedding
            grad_weight, used_indices = grads

            for idx in used_indices
                param_row = view(layer.weight, :, idx)
                grad_row = view(grad_weight, :, idx)

                key = (objectid(layer.weight), idx)
                m, v, t = get!(opt.state, key, (zeros(Float32, size(param_row)), zeros(Float32, size(param_row)), 0))
                t += 1
                m .= opt.β1 .* m .+ (1 - opt.β1) .* grad_row
                v .= opt.β2 .* v .+ (1 - opt.β2) .* (grad_row .^ 2)
                m̂ = m ./ (1 - opt.β1^t + Float32(1e-8))
                ṽ = v ./ (1 - opt.β2^t + Float32(1e-8))
                # Learning rate scheduling
                current_lr = opt.lr * Float32(0.9)^(div(t, 1000))
                param_row .-= current_lr .* m̂ ./ (sqrt.(ṽ) .+ opt.ϵ)
                opt.state[key] = (m, v, t)
            end
        elseif layer isa BatchNorm1D
            grad_gamma, grad_beta = grads
            for (param, grad) in zip((layer.gamma, layer.beta), (grad_gamma, grad_beta))
                key = objectid(param)
                m, v, t = get!(opt.state, key, (zeros(Float32, size(param)), zeros(Float32, size(param)), 0))
                t += 1
                m .= opt.β1 .* m .+ (1 - opt.β1) .* grad
                v .= opt.β2 .* v .+ (1 - opt.β2) .* (grad .^ 2)
                m̂ = m ./ (1 - opt.β1^t + Float32(1e-8))
                ṽ = v ./ (1 - opt.β2^t + Float32(1e-8))
                # Learning rate scheduling
                current_lr = opt.lr * Float32(0.9)^(div(t, 1000))
                param .-= current_lr .* m̂ ./ (sqrt.(ṽ) .+ opt.ϵ)
                opt.state[key] = (m, v, t)
            end
        end
    end
end



# Modify params to work with new Dense layer
function params(model)
    ps = []
    if model isa Dense
        push!(ps, model.W)
        push!(ps, model.b)
    elseif model isa Chain
        for layer in model.layers
            append!(ps, params(layer))
        end
    end
    return ps
end

function setparams!(model, new_params)
    idx = 1
    if model isa Dense
        model.W .= new_params[idx]; idx += 1
        model.b .= new_params[idx]; idx += 1
    elseif model isa Chain
        for layer in model.layers
            if layer isa Dense
                layer.W .= new_params[idx]; idx += 1
                layer.b .= new_params[idx]; idx += 1
            end
        end
    end
end

binarycrossentropy(ŷ, y) = AD.binarycrossentropy(ŷ, y)

# DataLoader definition
struct DataLoader
    data::Tuple{Array{Float32,2}, Array{Float32,1}}
    batchsize::Int
    shuffle::Bool
    indices::Vector{Int}
end

function DataLoader(data::Tuple{<:AbstractArray, <:AbstractArray}; batchsize::Int=64, shuffle::Bool=true)
    X, y = data
    @assert size(X, 2) == length(y) "Mismatched number of samples in X and y"

    # Convert to Float32 automatically
    X_float32 = Float32.(X)
    y_float32 = Float32.(y)

    idx = shuffle ? randperm(size(X_float32, 2)) : collect(1:size(X_float32, 2))

    # Store original indices for leakage detection
    original_indices = collect(1:size(X_float32, 2))

    return DataLoader((X_float32, y_float32), batchsize, shuffle, idx)
end

function Base.iterate(dl::DataLoader, state::Int=1)
    if state > length(dl.indices)
        return nothing
    end

    end_idx = min(state + dl.batchsize - 1, length(dl.indices))
    batch_indices = dl.indices[state:end_idx]

    X, y = dl.data
    batch_X = X[:, batch_indices]
    batch_y = y[batch_indices]

    return (batch_X, batch_y), end_idx + 1
end

Base.length(dl::DataLoader) = cld(length(dl.indices), dl.batchsize)

function clip_gradients!(grads, clip_value=Float32(1.0))
    for grad in grads
        if grad !== nothing
            if isa(grad, Tuple) && length(grad) == 2 &&
               isa(grad[1], AbstractArray{Float32}) && isa(grad[2], AbstractArray{Float32})
                gW, gb = grad
                norm = sqrt(sum(gW.^2) + sum(gb.^2))
                if norm > clip_value
                    scale = clip_value / (norm + Float32(1e-6))
                    gW .*= scale
                    gb .*= scale
                end
            end
        end
    end
end

function loss(model, x, y)
    y_pred = vec(model(x))
    ϵ = Float32(1e-7)
    ŷ = clamp.(y_pred, ϵ, Float32(1.0) - ϵ)

    # Binary cross entropy loss - more efficient computation
    bce_loss = -sum(y .* log.(ŷ) .+ (Float32(1.0) .- y) .* log.(Float32(1.0) .- ŷ)) / length(y)

    # L2 regularization - only compute for Dense layers, avoid repeated allocations
    l2_reg = Float32(0.0)
    for layer in model.layers
        if layer isa Dense
            l2_reg += sum(layer.W.^2) + sum(layer.b.^2)
        elseif layer isa Conv1D
            l2_reg += sum(layer.weight.^2) + sum(layer.bias.^2)
        end
    end

    return bce_loss + Float32(0.0001) * l2_reg
end

function grad(model, x, y)
    y_pred = vec(model(x))
    # More efficient gradient computation
    grad_pred = (y_pred .- y) ./ max.(y_pred .* (Float32(1.0) .- y_pred), Float32(1e-7))
    grad_pred ./= length(y)
    return reshape(grad_pred, :, 1)
end

function train_model(model, dataset, test_X, test_y, opt, epochs)
    for epoch in 1:epochs
        # Set training mode
        set_train_mode!(model)

        total_loss = Float32(0.0)
        total_acc = Float32(0.0)
        num_samples = 0

        t = @elapsed begin
            for (x, y) in dataset
                # Forward pass
                y_pred = model(x)

                # Compute loss and accuracy more efficiently
                current_loss = loss(model, x, y)
                current_acc = mean((vec(y_pred) .> 0.5) .== (y .> 0.5))

                # Backward pass - compute gradients
                grad_output = grad(model, x, y)
                grads_cache = backward!(model, grad_output)

                # Clip gradients to prevent exploding gradients
                clip_gradients!(grads_cache, Float32(1.0))

                # Update parameters
                update!(model, grads_cache, opt)

                # Accumulate metrics
                total_loss += current_loss
                total_acc += current_acc
                num_samples += 1
            end
        end

        # Calculate average metrics
        train_loss = total_loss / num_samples
        train_acc = total_acc / num_samples

        # Set evaluation mode for testing
        set_eval_mode!(model)

        # Test metrics - compute once
        test_pred = vec(model(test_X))
        test_acc = mean((test_pred .> 0.5) .== (test_y .> 0.5))
        test_loss = loss(model, test_X, test_y)

        # Print progress
        println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.2f, a: %.2f) \tTest: (l: %.2f, a: %.2f)",
            epoch, t, train_loss, train_acc, test_loss, test_acc))
    end
end

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
        ones(Float32, channels)
    )
end

function (bn::BatchNorm1D)(x::Array{Float32, 3})
    bn.last_input = x
    L, C, N = size(x)

    if bn.is_training
        # Compute batch statistics
        bn.batch_mean = vec(mean(x, dims=(1, 3)))  # Mean over spatial and batch dimensions
        bn.batch_var = vec(var(x, dims=(1, 3), corrected=false))  # Variance over spatial and batch dimensions

        # Update running statistics
        bn.running_mean .= (1 - bn.momentum) .* bn.running_mean .+ bn.momentum .* bn.batch_mean
        bn.running_var .= (1 - bn.momentum) .* bn.running_var .+ bn.momentum .* bn.batch_var

        # Normalize using batch statistics
        x_norm = (x .- reshape(bn.batch_mean, 1, :, 1)) ./ sqrt.(reshape(bn.batch_var, 1, :, 1) .+ bn.eps)
    else
        # Use running statistics for inference
        x_norm = (x .- reshape(bn.running_mean, 1, :, 1)) ./ sqrt.(reshape(bn.running_var, 1, :, 1) .+ bn.eps)
    end

    # Scale and shift
    bn.last_output = reshape(bn.gamma, 1, :, 1) .* x_norm .+ reshape(bn.beta, 1, :, 1)

    return bn.last_output
end

function backward!(bn::BatchNorm1D, grad_output::Array{Float32, 3})
    if !bn.is_training
        # For inference, just pass through
        return grad_output
    end

    x = bn.last_input
    L, C, N = size(x)

    # Compute gradients
    x_centered = x .- reshape(bn.batch_mean, 1, :, 1)
    std_inv = 1.0 ./ sqrt.(bn.batch_var .+ bn.eps)

    # Gradients for gamma and beta
    grad_gamma = vec(sum(grad_output .* (x_centered .* reshape(std_inv, 1, :, 1)), dims=(1, 3)))
    grad_beta = vec(sum(grad_output, dims=(1, 3)))

    # Gradient for input
    grad_input = grad_output .* reshape(bn.gamma .* std_inv, 1, :, 1)

    return grad_gamma, grad_beta, grad_input
end

# Define Layer union type after all structs are defined
const Layer = Union{Dense, Dropout, Chain, Embedding, Conv1D, MaxPool1D, BatchNorm1D, typeof(flatten)}

# Add evaluation mode functions
function set_eval_mode!(model::Chain)
    for layer in model.layers
        if layer isa Dropout
            layer.is_training = false
        elseif layer isa BatchNorm1D
            layer.is_training = false
        end
    end
end

function set_train_mode!(model::Chain)
    for layer in model.layers
        if layer isa Dropout
            layer.is_training = true
        elseif layer isa BatchNorm1D
            layer.is_training = true
        end
    end
end

end # module
