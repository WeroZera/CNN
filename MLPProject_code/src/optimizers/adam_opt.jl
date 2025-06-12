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