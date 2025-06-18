module AD

export ADValue, grad, unwrap, binarycrossentropy

import Base: +, -, *, /, exp, clamp, zero, one

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


zero(::Type{ADValue}) = ADValue(0.0)
one(::Type{ADValue}) = ADValue(1.0)


function sum(arr::AbstractArray{ADValue})
    result = zero(ADValue)
    for x in arr
        result += x
    end
    return result
end


function sigmoid(x::ADValue)
    s = 1 / (1 + exp(-x))
    return s  
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

function grad(f, params)
    ad_params = map(p -> map(x -> ADValue(x), p), params)
    out = f(ad_params...)
    backward(out)
    grads = map(p -> map(x -> x.grad, p), ad_params)
    return grads, out.value
end

end # module
