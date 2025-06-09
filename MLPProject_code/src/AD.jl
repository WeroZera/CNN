module AD

export ADValue, grad, unwrap, binarycrossentropy

import Base: +, -, *, /, zero, one, sum, exp, clamp

struct ADValue
    value::Float64
    grad::Float64
    ADValue(v::Number, g::Number) = new(float(v), float(g))
end

ADValue(v::Number) = ADValue(v, 0.0)

unwrap(x::ADValue) = x.value
unwrap(x) = x

# Operator +
+(a::ADValue, b::ADValue) = ADValue(a.value + b.value, a.grad + b.grad)
+(a::ADValue, b::Number) = ADValue(a.value + b, a.grad)
+(a::Number, b::ADValue) = ADValue(a + b.value, b.grad)

# Operator -
-(a::ADValue, b::ADValue) = ADValue(a.value - b.value, a.grad - b.grad)
-(a::ADValue, b::Number) = ADValue(a.value - b, a.grad)
-(a::Number, b::ADValue) = ADValue(a - b.value, -b.grad)

# Operator *
*(a::ADValue, b::ADValue) = ADValue(a.value * b.value, a.grad * b.value + a.value * b.grad)
*(a::ADValue, b::Number) = ADValue(a.value * b, a.grad * b)
*(a::Number, b::ADValue) = ADValue(a * b.value, a * b.grad)

# Operator /
/(a::ADValue, b::ADValue) = ADValue(a.value / b.value, (a.grad * b.value - a.value * b.grad) / b.value^2)
(/)(a::ADValue, b::Number) = ADValue(a.value / b, a.grad / b)
(/)(a::Number, b::ADValue) = ADValue(a / b.value, (-a * b.grad) / b.value^2)

# Operator unary -
-(a::ADValue) = ADValue(-a.value, -a.grad)

# zero & one
zero(::Type{ADValue}) = ADValue(0.0, 0.0)
one(::Type{ADValue}) = ADValue(1.0, 0.0)

# exp
exp(a::ADValue) = ADValue(exp(a.value), exp(a.value) * a.grad)

# clamp
clamp(x::ADValue, lo, hi) = ADValue(clamp(x.value, lo, hi), x.grad * (x.value > lo && x.value < hi ? 1.0 : 0.0))

# sum – tylko dla Array{ADValue}
sum(arr::AbstractArray{ADValue}) = foldl((a, b) -> a + b, arr)

import Base: copy
function copy(x::ADValue)
    ADValue(x.value, x.grad)
end

# Funkcja sigmoid
function sigmoid(x::ADValue)
    sig = 1 / (1 + exp(-x))
    # pochodna sigmoid(x) = sigmoid(x)*(1-sigmoid(x))
    return sig
end

# Binary cross entropy function
function binarycrossentropy(ŷ, y)
    ϵ = 1e-7
    ŷ_clamped = clamp.(ŷ, ϵ, 1.0 - ϵ)
    return -mean(y .* log.(ŷ_clamped) .+ (1.0 .- y) .* log.(1.0 .- ŷ_clamped))
end

import Base: size
size(x::ADValue) = ()

import Base: length
length(x::ADValue) = 1

# grad – liczy gradient po parametrach modelu względem funkcji f
function grad(f, params)
    # Zamieniamy wszystkie parametry na ADValue, liczymy pochodną po każdym z nich
    orig_params = [copy(p) for p in params]
    grads = [zeros(Float64, size(p)) for p in params]
    l = 0.0

    for i in eachindex(params)
        p = params[i]
        if p isa ADValue
            # Handle scalar ADValue case
            ad_params = [copy(x) for x in params]
            ad_params[i] = ADValue(p.value, 1.0)
            for j in 1:length(params)
                if j != i
                    if params[j] isa ADValue
                        ad_params[j] = ADValue(params[j].value, 0.0)
                    else
                        ad_params[j] = map(x -> ADValue(x, 0.0), params[j])
                    end
                end
            end
            lval = f(ad_params...)
            grads[i] = lval.grad
            if i == 1
                l = lval.value
            end
        else
            # Handle array case
            sz = size(p)
            g = zeros(Float64, sz)
            for idx in CartesianIndices(sz)
                ad_params = [copy(x) for x in params]
                ad_params[i][idx] = ADValue(p[idx], 1.0)
                for j in 1:length(params)
                    if j != i
                        if params[j] isa ADValue
                            ad_params[j] = ADValue(params[j].value, 0.0)
                        else
                            ad_params[j] = map(x -> ADValue(x, 0.0), params[j])
                        end
                    end
                end
                lval = f(ad_params...)
                g[idx] = lval.grad
                if i == 1 && idx == CartesianIndex((ones(Int, ndims(p)))...)
                    l = lval.value
                end
            end
            grads[i] = g
        end
    end
    return grads, l
end

end # module
