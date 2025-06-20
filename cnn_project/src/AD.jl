module AD

export ADValue, grad, unwrap, binarycrossentropy, clip_gradients!, loss

import Base: +, -, *, /, zero, one, sum, exp, clamp

mutable struct ADValue
    value::Float64
    grad::Float64
    parents::Vector{Tuple{ADValue, Function}} 

    ADValue(v::Float64) = new(v, 0.0, [])
end

unwrap(x::ADValue) = x.value
unwrap(x) = x

# Add 
+(a::ADValue, b::ADValue) = begin
    out = ADValue(a.value + b.value)
    push!(out.parents, (a, Δ -> Δ))
    push!(out.parents, (b, Δ -> Δ))
    return out
end
+(a::ADValue, b::Number) = a + ADValue(b)
+(a::Number, b::ADValue) = ADValue(a) + b

# Subtract 
-(a::ADValue, b::ADValue) = begin
    out = ADValue(a.value - b.value)
    push!(out.parents, (a, Δ -> Δ))
    push!(out.parents, (b, Δ -> -Δ))
    return out
end
-(a::ADValue, b::Number) = a - ADValue(b)
-(a::Number, b::ADValue) = ADValue(a) - b

# Multiply
*(a::ADValue, b::ADValue) = begin
    out = ADValue(a.value * b.value)
    push!(out.parents, (a, Δ -> Δ * b.value))
    push!(out.parents, (b, Δ -> Δ * a.value))
    return out
end
*(a::ADValue, b::Number) = a * ADValue(b)
*(a::Number, b::ADValue) = ADValue(a) * b

# Divide
/(a::ADValue, b::ADValue) = begin
    out = ADValue(a.value / b.value)
    push!(out.parents, (a, Δ -> Δ / b.value))
    push!(out.parents, (b, Δ -> -Δ * a.value / b.value^2))
    return out
end
/(a::ADValue, b::Number) = a / ADValue(b)
/(a::Number, b::ADValue) = ADValue(a) / b


-(a::ADValue) = begin
    out = ADValue(-a.value)
    push!(out.parents, (a, Δ -> -Δ))
    return out
end


exp(a::ADValue) = begin
    out = ADValue(exp(a.value))
    push!(out.parents, (a, Δ -> Δ * out.value))
    return out
end


clamp(x::ADValue, lo, hi) = begin
    v = clamp(x.value, lo, hi)
    out = ADValue(v)
    grad_fn = Δ -> (x.value > lo && x.value < hi) ? Δ : 0.0
    push!(out.parents, (x, grad_fn))
    return out
end

# Define activation functions
relu(x) = x > 0 ? x : Float32(0)
relu(x::AbstractArray) = max.(x, Float32(0))

# LeakyReLU - often better than ReLU
leakyrelu(x, α=Float32(0.01)) = x > 0 ? x : α * x
leakyrelu(x::AbstractArray, α=Float32(0.01)) = max.(x, α .* x)

# ELU - exponential linear unit
elu(x, α=Float32(1.0)) = x > 0 ? x : α * (exp(x) - 1)
elu(x::AbstractArray, α=Float32(1.0)) = x .> 0 ? x : α .* (exp.(x) .- 1)

sigmoid(x) = Float32(1) / (Float32(1) + exp(-x))
sigmoid(x::AbstractArray) = Float32(1) ./ (Float32(1) .+ exp.(-x))

softmax(xs) = begin
    exps = exp.(xs .- maximum(xs))
    exps ./ sum(exps)
end


function binarycrossentropy(ŷ, y)
    ϵ = 1e-7
    ŷ_clamped = clamp.(ŷ, ϵ, 1.0 - ϵ)
    return -mean(y .* log.(ŷ_clamped) .+ (1.0 .- y) .* log.(1.0 .- ŷ_clamped))
end


function backward(output::ADValue)
    output.grad = 1.0
    visited = Set{ADValue}()
    function _backward(node::ADValue)
        if node ∈ visited
            return
        end
        union!(visited, [node])
        for (parent, grad_fn) in node.parents
            parent.grad += grad_fn(node.grad)
            _backward(parent)
        end
    end
    _backward(output)
end

function loss(model, x, y)
    y_pred = vec(model(x))
    ϵ = 1e-7
    ŷ = clamp.(y_pred, ϵ, 1.0 - ϵ)
    return -sum(y .* log.(ŷ) .+ (1 .- y) .* log.(1 .- ŷ)) / length(y)
end

function grad(model, x, y)
    y_pred = vec(model(x))
    ϵ = 1e-7
    y_pred_clamped = clamp.(y_pred, ϵ, 1.0 - ϵ)
    grad_pred = (y_pred_clamped .- y) ./ max.(y_pred_clamped .* (1 .- y_pred_clamped), ϵ)
    return reshape(grad_pred ./ length(y), :, 1)
end

function clip_gradients!(grads, clip_value=1.0f0)
    for grad in grads
        if grad isa Tuple{<:AbstractArray{Float32}, <:AbstractArray{Float32}}
            gW, gb = grad
            norm = sqrt(sum(gW.^2) + sum(gb.^2))
            if norm > clip_value
                scale = clip_value / (norm + 1e-6f0)
                gW .*= scale
                gb .*= scale
            end
        end
    end
end

end # module