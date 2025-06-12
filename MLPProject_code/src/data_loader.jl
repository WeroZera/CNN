struct DataLoader
    data::Tuple{Array{Float32,2}, Array{Float32,1}}
    batchsize::Int
    shuffle::Bool
    indices::Vector{Int}
end

function DataLoader(data::Tuple{<:AbstractArray, <:AbstractArray}; batchsize::Int=64, shuffle::Bool=true)
    X, y = data
    @assert size(X, 2) == length(y) "Mismatched number of samples in X and y"

    X_float32 = Float32.(X)
    y_float32 = Float32.(y)

    idx = shuffle ? randperm(size(X_float32, 2)) : collect(1:size(X_float32, 2))
    
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