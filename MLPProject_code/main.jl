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

accuracy(m, x, y) = mean((vec(m(x)) .> 0.5) .== (y .> 0.5))

# Use a smaller learning rate
global opt = NN.Adam(Float32(0.0001))  # Reduced learning rate
epochs = 5  # Increased epochs for early stopping
patience = 5  # Number of epochs to wait for improvement

# Run training
model = NN.train_model(model, dataset, X_test, y_test, opt, epochs, patience)
