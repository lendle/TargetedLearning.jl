module LReg

using GLM

using NumericExtensions, NumericFuns, Devectorize

import StatsBase: predict, coef
import Distributions: log1pexp

export AbstractLR, LR, SSLR, lreg, predict

abstract AbstractLR

type LR{T<:FloatingPoint} <: AbstractLR
    β::Vector{T}
end

type SSLR{T <: FloatingPoint} <: AbstractLR
    β::Vector{T}
    idx::AbstractVector
end

function linpred(lr::AbstractLR, newX; offset=nothing)
    p = newX * lr.β
    isa(offset, Nothing) || add!(p, offset)
    p
end

predict(lr::AbstractLR, newX; offset=nothing) = map1!(LogisticFun(), linpred(lr, newX, offset=offset))

function lreg(x, y; wts=ones(y), offset=similar(y,0), subset=nothing)
    if isa(subset, Nothing)
        return LR(coef(fit(GeneralizedLinearModel, x, y, Binomial(); wts=wts, offset=offset)))
    else
        subset=sort(subset)
        tempx = subset == 1:size(x,2) || collect(1:size(x,2)) == collect(subset)? x : x[:, subset]
        β = zeros(eltype(x), size(x, 2))
        β[subset] = coef(fit(GeneralizedLinearModel, tempx, y, Binomial(); wts=wts, offset=offset))
        return SSLR(β, subset)
    end
end

# type Loss <: Functor{2} end
# NumericExtensions.evaluate(::Loss, y, xb) =
#     y == one(y)? log1pexp(-xb) :
#     y == zero(y)? log1pexp(xb) :
#     y * log1pexp(-xb) + (one(y)-y) * log1pexp(xb)

end
