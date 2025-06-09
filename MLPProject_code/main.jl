using JLD2
using Printf
using Statistics
using LinearAlgebra

include("src/AD.jl")
include("src/NN.jl")

# Load the data directly using the available keys
file = load("data/imdb_dataset_prepared.jld2")
X_train = Float32.(Matrix(file["X_train"]))
y_train = vec(Float32.(file["y_train"]))
X_test  = Float32.(Matrix(file["X_test"]))
y_test  = vec(Float32.(file["y_test"]))

dataset = NN.DataLoader((X_train, y_train), batchsize=64, shuffle=true)

# Create model with dropout and weight decay
model = NN.Chain(
    NN.Dense(size(X_train, 1), 64, NN.relu, weight_decay=Float32(0.0001)),
    NN.Dropout(Float32(0.3)),
    NN.Dense(64, 32, NN.relu, weight_decay=Float32(0.001)),
    NN.Dropout(Float32(0.3)),
    NN.Dense(32, 1, NN.sigmoid, weight_decay=Float32(0.001))
)

# Binary cross-entropy loss with direct gradient computation
function loss_and_grad(model, x, y)
    y_pred = vec(model(x))  # convert (batch_size, 1) → (batch_size)

    ϵ = Float32(1e-7)
    ŷ = clamp.(y_pred, ϵ, 1f0 - ϵ)
    loss = -mean(y .* log.(ŷ) .+ (1f0 .- y) .* log.(1f0 .- ŷ))

    grad_pred = (y_pred .- y) ./ (y_pred .* (1f0 .- y_pred))
    grad_pred ./= length(y)
    grad_pred = reshape(grad_pred, :, 1)  # convert back to (batch_size, 1) for backward pass

    return loss, grad_pred
end

accuracy(m, x, y) = mean((vec(m(x)) .> 0.5) .== (y .> 0.5))

# Use a smaller learning rate
global opt = NN.Adam(Float32(0.0001))  # Reduced learning rate
epochs = 5  # Increased epochs for early stopping
patience = 5  # Number of epochs to wait for improvement

# Training loop with early stopping
function train_model()
    local best_val_loss = Inf
    local patience_counter = 0
    local best_model_state = nothing
    local current_model = deepcopy(model)  # Work with a copy of the model

    for epoch in 1:epochs
        total_loss = 0.0
        total_acc = 0.0
        num_samples = 0
        batch_count = 0

        # Set dropout to training mode
        for layer in current_model.layers
            if layer isa NN.Dropout
                layer.is_training = true
            end
        end

        t = @elapsed begin
            for (x, y) in dataset
                batch_count += 1

                # Compute loss and gradients
                loss_val, grad_output = loss_and_grad(current_model, x, y)

                # Backward pass
                grads_cache = NN.backward!(current_model, reshape(grad_output, :, 1))

                # Update parameters
                NN.clip_gradients!(grads_cache, 5.0f0)  # lub 1.0f0
                NN.update!(current_model, grads_cache, opt)


                total_loss += loss_val
                total_acc += accuracy(current_model, x, y)
                num_samples += 1
            end
        end

        # Set dropout to evaluation mode for validation
        for layer in current_model.layers
            if layer isa NN.Dropout
                layer.is_training = false
            end
        end

        train_loss = total_loss / num_samples
        train_acc = total_acc / num_samples
        test_acc = accuracy(current_model, X_test, y_test)
        test_loss = loss_and_grad(current_model, X_test, y_test)[1]

        println(@sprintf("Epoch: %d \tTrain: (l: %.4f, a: %.4f) \tTest: (l: %.4f, a: %.4f)",
            epoch, train_loss, train_acc, test_loss, test_acc))

        # Early stopping check
        if test_loss < best_val_loss
            best_val_loss = test_loss
            patience_counter = 0
            # Save best model state
            best_model_state = deepcopy(current_model)
        else
            patience_counter += 1
            if patience_counter >= patience
                println("\nEarly stopping triggered after $epoch epochs")
                # Restore best model
                current_model = best_model_state
                break
            end
        end
    end

    # Set dropout to evaluation mode for final evaluation
    for layer in current_model.layers
        if layer isa NN.Dropout
            layer.is_training = false
        end
    end

    return current_model
end

# Run training
global model = train_model()
