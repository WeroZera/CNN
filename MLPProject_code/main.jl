using JLD2
using Printf
using Statistics
using LinearAlgebra

include("src/AD.jl")
include("src/NN.jl")

file = load("data/imdb_dataset_prepared.jld2")
X_train = Int.(file["X_train"])  # Use integers for embedding indices
y_train = vec(Float32.(file["y_train"]))
X_test  = Int.(file["X_test"])   # Use integers for embedding indices
y_test  = vec(Float32.(file["y_test"]))
embeddings = file["embeddings"]
vocab = file["vocab"]

embedding_dim = size(embeddings, 1)  # 50

# DataLoader expects (features, samples) shape, so transpose if needed
if size(X_train, 1) < size(X_train, 2)
    X_train = X_train
else
    X_train = X_train'
end
if size(X_test, 1) < size(X_test, 2)
    X_test = X_test
else
    X_test = X_test'
end

# DataLoader expects Matrix{Float32}, so convert after integer indexing
X_train_f32 = Float32.(X_train)
X_test_f32 = Float32.(X_test)

dataset = NN.DataLoader((X_train_f32, y_train), batchsize=32, shuffle=true)

# Calculate dimensions:
# Input: (130, batch_size) -> Embedding: (50, 130, batch_size) -> Permute: (130, 50, batch_size)
# Conv1D(3, 50, 8): (130-3+1, 8, batch_size) = (128, 8, batch_size)
# MaxPool1D(8): (128/8, 8, batch_size) = (16, 8, batch_size)
# Flatten: (16*8, batch_size) = (128, batch_size)

model = NN.Chain(
    x -> Int.(x),
    NN.Embedding(length(vocab), embedding_dim, embeddings),
    x -> permutedims(x, (2, 1, 3)),
    NN.Conv1D(3, embedding_dim, 16, NN.relu),
    NN.MaxPool1D(8),
    NN.flatten,
    NN.Dropout(0.3),
    NN.Dense(256, 64, NN.relu),
    NN.Dropout(0.3),
    NN.Dense(64, 1, NN.sigmoid)
);

accuracy(m, x, y) = mean((vec(m(x)) .> 0.5) .== (y .> 0.5))

opt = NN.Adam(Float32(0.001))
epochs = 5

NN.train_model(model, dataset, X_test_f32, y_test, opt, epochs)
