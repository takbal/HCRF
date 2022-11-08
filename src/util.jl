# numerically stable log( exp(x) + exp(y) )
function logaddexp(x::Number, y::Number)
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
function logsumexp(x::AbstractArray{I}) where {I<:Number}
    m = maximum(x)
    r = 0.0
    for i in eachindex(x)
        r += exp(x[i] - m)
    end
    return log(r) + m
end
