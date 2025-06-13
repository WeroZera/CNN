mutable struct Embedding
    vocab_size::Int
    embedding_dim::Int
    weight::Array{Float32, 2}
    last_indices::Vector{Int}
end

function Embedding(vocab_size::Int, embedding_dim::Int, weights::Array{<:Real,2})
    Embedding(vocab_size, embedding_dim, Float32.(weights), Int[])
end

function (e::Embedding)(x::AbstractArray{Int})
    inds = vec(x) 
    e.last_indices = inds
    emb = e.weight[:, inds]
    return reshape(emb, e.embedding_dim, size(x)...)
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