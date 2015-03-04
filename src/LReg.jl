module LReg

using Docile
@document

"""This module wraps some functionality from [GLM.jl](https://github.com/JuliaStats/GLM.jl)
for logistic regression"""
LReg

using GLM

using NumericExtensions, NumericFuns, StatsBase

import StatsBase: predict
import Distributions: log1pexp

export LR, lreg, predict, linpred

"""The `LR` type contains the coefficent vector of a logistic regression fit, as well as indexes of
included columns in the design matrix.
"""
immutable LR{T<:FloatingPoint}
    β::Vector{T}
    idx::AbstractVector{Int}
    fitwithoffset::Bool
end

"""
Returns the linear component of predicted values from a logistic regression fit with a new X matrix

** Arguments **

* `lr` - an object of type `LR`
* `newx` - matrix of new covariates to predict on. Should be the same dimension as the design matrix used to fit `lr`

** Keyword Arguments **

* `offset` - offsets. The length should be the same as `size(newx, 1)` of `lr` was fit with an offset, or 0 otherwise.
"""
function linpred(lr::LR, newx; offset=Array(eltype(newx), 0))
    if size(newx, 2) == 1
        newx = reshape(newx, size(newx, 1), 1)
    end
    p = newx * lr.β
    if lr.fitwithoffset
        length(offset) == size(newx, 1) || error(ArgumentError("fit with offset, 'offset' kw arg should have length size(newx, 1)"))
        add!(p, offset)
    else
        length(offset) == 0 || error(ArgumentError("not fit with offset, 'offset' kw arg should have length 0"))
    end
    p
end

"""
Returns the predicted on the probability scale values from a logistic regression fit with a new X matrix

** Arguments **

* `lr` - an object of type `LR`
* `newx` - matrix of new covariates to predict on. Should be the same dimension as the design matrix used to fit `lr`

** Keyword Arguments

* `offset` - offsets. The length should be the same as `size(newx, 1)` of `lr` was fit with an offset, or 0 otherwise.
"""
predict(lr::LR, newx; offset=Array(eltype(newx), 0)) = map1!(LogisticFun(), linpred(lr, newx, offset=offset))

"""
Fits a logistic regression model

** Arguments **

* `x` - design matrix
* `y` - response vector. Should have the same length as `size(x, 1)`

** Keyword Arguments **

* `wts` - weight vector. Defaults to all 1s.
* `offset` - offset vector. Defaults to a vector of length 0 for no offset.
* `subset` - column indexes for `x` that should be included in the fit. Defaults to all columns.
* `convTol` - convergence criterion for relative change in deviance. Defaults to 1.0e-6.

** Details **

An intercept is not included by default. If you want one, add a column of ones to your design matrix.
If you do that, don't forget to add the column in the same place to `newx` when you call predict.

`subset` is useful if you want to include only some columns of the design matrix but you want to call `predict`
with a `newx` matrix with the same number of columns as `x`. The coefficients corresponding to columsn of `x` which
are not used in the fit are set to zero. If you would like to call `predict` with a `newx` matrix that includes
only the columns that you fit on, you should subset `x` yourself before calling `lreg`.
"""
function lreg(x, y; wts=ones(y), offset=similar(y,0), subset=1:size(x,2), convTol=1.0e-6)
    if size(x, 2) == 1
        x = reshape(x, size(x, 1), 1)
    end
    subset=sort(subset)
    fitwithoffset = length(offset) > 0

    if subset == 1:size(x,2) || collect(1:size(x,2)) == subset
        subset = 1:size(x,2)
        return LR(coef(fit(GeneralizedLinearModel, x, y, Binomial(); wts=wts, offset=offset, convTol=convTol)), subset, fitwithoffset)
    else
        tempx = x[:, subset]
        β = zeros(eltype(x), size(x, 2))
        β[subset] = coef(fit(GeneralizedLinearModel, tempx, y, Binomial(); wts=wts, offset=offset, convTol=convTol))
        return LR(β, subset, fitwithoffset)
    end
end

type Loss <: Functor{2} end
NumericExtensions.evaluate(::Loss, y, xb) =
    y == one(y)? log1pexp(-xb) :
    y == zero(y)? log1pexp(xb) :
    y * log1pexp(-xb) + (one(y)-y) * log1pexp(xb)

end
