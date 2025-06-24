mutable struct Embedding
    vocab_size::Int
    embedding_dim::Int
    weight::Array{Float32, 2}
    last_indices::Vector{Int}
    grad_weight::Array{Float32, 2}
end

function Embedding(vocab_size::Int, embedding_dim::Int, weights::Array{<:Real,2})
    Embedding(
        vocab_size,
        embedding_dim,
        Float32.(weights),
        Int[],
        zeros(Float32, embedding_dim, vocab_size)
    )
end

function (e::Embedding)(x::AbstractArray{Int})
    inds = vec(x)
    e.last_indices = inds
    emb = e.weight[:, inds]
    return reshape(emb, e.embedding_dim, size(x)...)
end

function backward!(e::Embedding, grad_output::Array{Float32, 2})
    fill!(e.grad_weight, 0f0)

    for (i, idx) in enumerate(e.last_indices)
        @inbounds e.grad_weight[:, idx] .+= grad_output[:, i]
    end

    unique_indices = unique(e.last_indices)
    for idx in unique_indices
        count = sum(j == idx for j in e.last_indices)
        if count > 1
            @inbounds e.grad_weight[:, idx] ./= count
        end
    end

    return e.grad_weight, unique_indices
end
