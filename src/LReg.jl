module LReg

using Docile
@document

"""This module wraps [GLM.jl](https://github.com/JuliaStats/GLM.jl)'s logistic regression"""
LReg

using GLM

using NumericExtensions, NumericFuns, Devectorize

import StatsBase: predict, coef
import Distributions: log1pexp

export LR, lreg, predict, linpred

immutable LR{T<:FloatingPoint}
    β::Vector{T}
    idx::AbstractVector{Int}
    fitwithoffset::Bool
end

function linpred(lr::LR, newX; offset=Array(eltype(newX), 0))
    if size(newX, 2) == 1
        newX = reshape(newX, size(newX, 1), 1)
    end
    p = newX * lr.β
    if lr.fitwithoffset
        length(offset) == size(newX, 1) || error(ArgumentError("fit with offset, 'offset' kw arg should have length size(newX, 1)"))
        add!(p, offset)
    else
        length(offset) == 0 || error(ArgumentError("not fit with offset, 'offset' kw arg should have length 0"))
    end
    p
end

predict(lr::LR, newX; offset=Array(eltype(newX), 0)) = map1!(LogisticFun(), linpred(lr, newX, offset=offset))

function lreg(x, y; wts=ones(y), offset=similar(y,0), subset=1:size(x,2))
    if size(x, 2) == 1
        x = reshape(x, size(x, 1), 1)
    end
    subset=sort(subset)
    fitwithoffset = length(offset) > 0
    
    if subset == 1:size(x,2) || collect(1:size(x,2)) == subset
        subset = 1:size(x,2)
        return LR(coef(fit(GeneralizedLinearModel, x, y, Binomial(); wts=wts, offset=offset)), subset, fitwithoffset)
    else
        tempx = x[:, subset]
        β = zeros(eltype(x), size(x, 2))
        β[subset] = coef(fit(GeneralizedLinearModel, tempx, y, Binomial(); wts=wts, offset=offset))
        return LR(β, subset, fitwithoffset)
    end
end

# type Loss <: Functor{2} end
# NumericExtensions.evaluate(::Loss, y, xb) =
#     y == one(y)? log1pexp(-xb) :
#     y == zero(y)? log1pexp(xb) :
#     y * log1pexp(-xb) + (one(y)-y) * log1pexp(xb)

end
