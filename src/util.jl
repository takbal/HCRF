# numerically stable log( exp(x) + exp(y) )
@inline @fastmath function logaddexp(x::Number, y::Number)
    if x == y
        # infs
        return x + log(2)
    else
        tmp = x - y
        if tmp > 0
            return x + log1p(exp(-tmp))
        elseif tmp <= 0
            return y + log1p(exp(tmp))
        else # nans
            return tmp
        end
    end
end

# numerically stable Σ log( exp(x_i) )
@inline @fastmath function logsumexp(x::AbstractArray{I}) where {I<:Number}
    m = maximum(x)
    r = 0.0
    for i in eachindex(x)
        r += exp(x[i] - m)
    end
    return log(r) + m
end

function equal_partition(n::Int64, parts::Int64)
    if n < parts
        return [ x:x for x in 1:n ]
    end
    starts = push!(Int64.(round.(1:n/parts:n)), n+1)
    return [ starts[i]:starts[i+1]-1 for i in 1:length(starts)-1 ]
end

function equal_partition(V::AbstractVector, parts::Int64)
    ranges = equal_partition(length(V), parts)
    return [ view(V,range) for range in ranges ]
end