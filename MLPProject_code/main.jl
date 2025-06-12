using JLD2
using Printf
using Statistics
using LinearAlgebra

include("src/AD.jl")
include("src/NN.jl")

# Enable threading if available
if Threads.nthreads() > 1
    println("Using $(Threads.nthreads()) threads")
end

file = load("data/imdb_dataset_prepared.jld2")
X_train = Int.(file["X_train"])  # Use integers for embedding indices
y_train = vec(Float32.(file["y_train"]))
X_test  = Int.(file["X_test"])   # Use integers for embedding indices
y_test  = vec(Float32.(file["y_test"]))
embeddings = file["embeddings"]
vocab = file["vocab"]

embedding_dim = size(embeddings, 1)  # 50


dataset = NN.DataLoader((X_train, y_train), batchsize=64, shuffle=true)

# Calculate dimensions:
# Input: (130, batch_size) -> Embedding: (50, 130, batch_size) -> Permute: (130, 50, batch_size)
# Conv1D(3, 50, 8): (130-3+1, 8, batch_size) = (128, 8, batch_size)
# MaxPool1D(8): (128/8, 8, batch_size) = (16, 8, batch_size)
# Flatten: (16*8, batch_size) = (128, batch_size)

model = NN.Chain(
    x -> Int.(x),
    NN.Embedding(length(vocab), embedding_dim, embeddings),
    x -> permutedims(x, (2, 1, 3)),
    NN.Conv1D(3, embedding_dim, 8, NN.relu),
    NN.MaxPool1D(8),
    NN.flatten,
    NN.Dense(128, 1, NN.sigmoid)
);

accuracy(m, x, y) = mean((vec(m(x)) .> 0.5) .== (y .> 0.5))

opt = NN.Adam(Float32(0.001))
epochs = 5

NN.train_model(model, dataset, X_test, y_test, opt, epochs)
