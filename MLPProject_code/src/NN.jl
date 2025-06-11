module NN

using ..AD
using Random
using LinearAlgebra
using Statistics
using Printf

export Dense, Chain, relu, sigmoid, softmax, binarycrossentropy, params, update!,
       DataLoader, Adam, Dropout, clip_gradients!, train_model,
       Embedding, Conv1D, MaxPool1D, flatten

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
end

mutable struct MaxPool1D
    pool_size::Int
end

flatten(x) = reshape(x, :, size(x, 3))

const Layer = Union{Dense, Dropout, Chain, Embedding, Conv1D, MaxPool1D, typeof(flatten)}

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
    weight = randn(Float32, k, c_in, c_out) * Float32(0.1)
    bias = zeros(Float32, c_out)
    Conv1D(k, c_in, c_out, weight, bias, activation)
end

function (c::Conv1D)(x::Array{Float32, 3})
    L, C, N = size(x)
    k = c.kernel_size
    out_len = L - k + 1
    y = Array{Float32}(undef, out_len, c.out_channels, N)

    kernels = reshape(c.weight, k * C, c.out_channels)

    for n in 1:N
        col = Array{Float32}(undef, k * C, out_len)
        for i in 1:out_len
            patch = x[i:i+k-1, :, n]
            col[:, i] = reshape(patch, k * C)
        end
        z = kernels' * col  # shape: (out_channels, out_len)
        z .+= c.bias[:, :]  # add bias to each channel
        y[:, :, n] .= c.activation.(z')  # z' -> (out_len, out_channels)
    end

    return y
end

function (p::MaxPool1D)(x::Array{Float32, 3})
    L, C, N = size(x)
    stride = p.pool_size
    out_len = Int(floor(L / stride))
    result = Array{Float32, 3}(undef, out_len, C, N)
    for n in 1:N, c in 1:C
        for i in 1:out_len
            result[i, c, n] = maximum(view(x, ((i-1)*stride+1):(i*stride), c, n))
        end
    end
    return result
end

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

function Dropout(p=Float32(0.5))
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
    # grad_output ma wymiary (embedding_dim, batch_size)
    grad_weight = zeros(Float32, size(e.weight))  # (embedding_dim, vocab_size)

    # Dla każdego indeksu w last_indices dodaj gradient
    for (i, idx) in enumerate(e.last_indices)
        grad_weight[:, idx] .+= grad_output[:, i]
    end

    return grad_weight
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

# Internal constructor

function Chain(layers::Layer...)
    layer_list = collect(layers)
    grad_cache = [nothing for _ in layer_list]
    Chain(layer_list, grad_cache)
end

# External constructor for varargs
function Chain(layers::Any...)
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
        elseif layer isa Embedding || layer isa Conv1D || layer isa MaxPool1D
            # For CNN layers, just pass through the gradient (no parameters to update)
            c.grad_cache[i] = nothing
        elseif isa(layer, Function)
            # For function layers (like permute, flatten, etc.), just pass through
            c.grad_cache[i] = nothing
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
                param .-= opt.lr .* m̂ ./ (sqrt.(ṽ) .+ opt.ϵ)
                opt.state[key] = (m, v, t)
            end

        elseif layer isa Embedding
            grad_weight, used_indices = grads

            for idx in used_indices
                param_row = view(layer.weight, :, idx)
                grad_row = view(grad_weight, :, idx)

                # Use a stable key per row
                key = (objectid(layer.weight), idx)
                m, v, t = get!(opt.state, key, (zeros(Float32, size(param_row)), zeros(Float32, size(param_row)), 0))
                t += 1
                m .= opt.β1 .* m .+ (1 - opt.β1) .* grad_row
                v .= opt.β2 .* v .+ (1 - opt.β2) .* (grad_row .^ 2)
                m̂ = m ./ (1 - opt.β1^t + Float32(1e-8))
                ṽ = v ./ (1 - opt.β2^t + Float32(1e-8))
                param_row .-= opt.lr .* m̂ ./ (sqrt.(ṽ) .+ opt.ϵ)
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
end

function DataLoader(data::Tuple{Array{Float32,2},Array{Float32,1}}; batchsize::Int=64, shuffle::Bool=true)
    X, y = data
    # Data is already in the correct shape (features × samples), no need to transpose
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

function clip_gradients!(grads, clip_value=Float32(1.0))
    for grad in grads
        if grad !== nothing
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

function loss(model, x, y)
    y_pred = vec(model(x))
    ϵ = Float32(1e-7)
    ŷ = clamp.(y_pred, ϵ, Float32(1.0) - ϵ)

    # Binary cross entropy loss
    bce_loss = -mean(y .* log.(ŷ) .+ (Float32(1.0) .- y) .* log.(Float32(1.0) .- ŷ))

    # L2 regularization
    l2_reg = Float32(0.0)
    for layer in model.layers
        if layer isa Dense
            l2_reg += sum(layer.W.^2) + sum(layer.b.^2)
        end
    end

    return bce_loss + Float32(0.0001) * l2_reg
end

function grad(model, x, y)
    y_pred = vec(model(x))
    grad_pred = (y_pred .- y) ./ (y_pred .* (Float32(1.0) .- y_pred))
    grad_pred ./= length(y)
    return reshape(grad_pred, :, 1)
end

function train_model(model, dataset, test_X, test_y, opt, epochs)
    for epoch in 1:epochs
        total_loss = Float32(0.0) # Or zero(Float32)
        total_acc = Float32(0.0)  # Or zero(Float32)
        num_samples = 0

        t = @elapsed begin
            for (x, y) in dataset
                # Forward pass
                y_pred = model(x)

                # Compute loss
                current_loss = loss(model, x, y)
                total_loss += current_loss
                # Compute accuracy
                acc = mean((vec(y_pred) .> 0.5) .== (y .> 0.5))
                total_acc += acc
                # Backward pass - compute gradients
                grad_output = grad(model, x, y)
                grads_cache = backward!(model, grad_output)
                # Clip gradients to prevent exploding gradients
                clip_gradients!(grads_cache, Float32(1.0))
                # Update parameters
                update!(model, grads_cache, opt)
                num_samples += 1
            end
        end

        train_loss = total_loss / num_samples
        train_acc = total_acc / num_samples
        test_acc = mean((vec(model(test_X)) .> 0.5) .== (test_y .> 0.5))
        test_loss = loss(model, test_X, test_y)

        println(@sprintf("Epoch: %d  (%.2fs)   Train: (l: %.4f, a: %.4f)   Test: (l: %.4f, a: %.4f)",
            epoch, t, train_loss, train_acc, test_loss, test_acc))
    end
end

end # module
