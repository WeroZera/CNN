#= Importy i załadowanie danych =#
using JLD2
using Printf
using Statistics
using LinearAlgebra

include("src/AD.jl")
include("src/NN.jl")
#= Przygotowanie danych =#
file = load("data/imdb_dataset_prepared.jld2")
X_train = Int.(file["X_train"])
y_train = vec(Float32.(file["y_train"]))
X_test  = Int.(file["X_test"])
y_test  = vec(Float32.(file["y_test"]))
embeddings = file["embeddings"]
vocab = file["vocab"]
embedding_dim = size(embeddings, 1)

dataset = NN.DataLoader((X_train, y_train), batchsize=64, shuffle=true)

model = NN.Chain(
    x -> Int.(x),
    NN.Embedding(length(vocab), embedding_dim, embeddings),
    x -> permutedims(x, (2, 1, 3)),
    NN.Conv1D(3, embedding_dim, 8, AD.relu),
    NN.MaxPool1D(8),
    NN.flatten,
    NN.Dense(128, 1, AD.sigmoid)
)

accuracy(m, x, y) = mean((vec(m(x)) .> 0.5) .== (y .> 0.5))

#= Trening modelu =#
opt = NN.Adam(Float32(0.001))
epochs = 5
NN.train_model(model, dataset, X_test, y_test, opt, epochs)
