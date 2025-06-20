module NN

import ..AD
using Random
using LinearAlgebra
using Statistics
using Printf
using Distributions


mutable struct Chain
    layers::Vector{Any}
    grad_cache::Vector{Any}
    shape_cache::Vector{Any}
end

function Chain(layers...)
    layer_list = collect(layers)
    grad_cache = [nothing for _ in layer_list]
    shape_cache = [nothing for _ in layer_list]
    Chain(layer_list, grad_cache, shape_cache)
end

function Chain(layers::Any...)
    layers_vec = collect(layers)
    grad_cache = [nothing for _ in layers_vec]
    shape_cache = [nothing for _ in layers_vec]
    Chain(layers_vec, grad_cache, shape_cache)
end


include("layers/dense.jl")
include("layers/embedding.jl")
include("layers/convolution.jl")
include("layers/flatten.jl")
include("layers/maxpool.jl")
include("layers/dropout.jl")
include("layers/batchnorm.jl")

include("optimizers/adam_opt.jl")
include("data_loader.jl")


const Layer = Union{Dense, Dropout, Chain, Embedding, Conv1D, MaxPool1D, BatchNorm1D, typeof(flatten)}


export Dense, Chain, binarycrossentropy, params, update!,
       DataLoader, Adam, Dropout, train_model,
       Embedding, Conv1D, MaxPool1D, flatten, BatchNorm1D,
       set_eval_mode!, set_train_mode!

const ADValue = AD.ADValue
binarycrossentropy(ŷ, y) = AD.binarycrossentropy(ŷ, y)


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
                current_lr = opt.lr * Float32(0.9)^(div(t, 1000))
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


function train_model(model, dataset, test_X, test_y, opt, epochs)
    for epoch in 1:epochs
        # Set training mode
        set_train_mode!(model)

        total_loss = Float32(0.0)
        total_acc = Float32(0.0)
        num_samples = 0
        mem_before = Base.gc_bytes()

        t = @elapsed begin
            for (x, y) in dataset
                # Forward pass
                y_pred = model(x)

                # Compute loss and accuracy more efficiently
                current_loss = AD.loss(model, x, y)
                current_acc = mean((vec(y_pred) .> 0.5) .== (y .> 0.5))

                # Backward pass - compute gradients
                grad_output = AD.grad(model, x, y)
                grads_cache = backward!(model, grad_output)

                # Clip gradients to prevent exploding gradients
                AD.clip_gradients!(grads_cache, Float32(1.0))

                # Update parameters
                update!(model, grads_cache, opt)

                # Accumulate metrics
                batch_size = length(y)
                total_loss += current_loss * batch_size
                total_acc += current_acc * batch_size
                num_samples += batch_size
            end
        end

        mem_after = Base.gc_bytes()
        mem_used_MB = (mem_after - mem_before) / (1024^2)

        # Calculate average metrics
        train_loss = total_loss / num_samples
        train_acc = total_acc / num_samples

        # Set evaluation mode for testing
        set_eval_mode!(model)

        # Test metrics - compute once
        test_pred = vec(model(test_X))
        test_acc = mean((test_pred .> 0.5) .== (test_y .> 0.5))
        test_loss = AD.loss(model, test_X, test_y)

        # Print progress
        println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.2f, a: %.2f) \tTest: (l: %.2f, a: %.2f) \tMemory allocated: %.3f MB",
            epoch, t, train_loss, train_acc, test_loss, test_acc, mem_used_MB))
    end
end






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
