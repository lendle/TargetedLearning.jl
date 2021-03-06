module LReg

"""This module wraps some functionality from [GLMNet.jl](https://github.com/simonster/GLMNet.jl)
for logistic regression."""
LReg

using GLMNet, Distributions

using StatsBase, StatsFuns

import StatsBase: predict
import Distributions: log1pexp

export LR, lreg, predict, linpred

"""The `LR` type contains the coefficent vector of a logistic regression fit, as well as indexes of
included columns in the design matrix.
"""
immutable LR{T<:AbstractFloat}
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
        length(offset) == size(newx, 1) || throw(ArgumentError("fit with offset, 'offset' kw arg should have length size(newx, 1)"))
        for i in 1:length(p) #NumericExtensions
          @inbounds p[i] += offset[i]
        end
    else
        length(offset) == 0 || throw(ArgumentError("not fit with offset, 'offset' kw arg should have length 0"))
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
predict(lr::LR, newx; offset=Array(eltype(newx), 0)) = logistic(linpred(lr, newx, offset=offset))

"""
Fits a logistic regression model

** Arguments **

* `x` - design matrix
* `y` - response vector. Should have the same length as `size(x, 1)`

** Keyword Arguments **

* `wts` - weight vector. Defaults to all 1s.
* `offset` - offset vector. Defaults to a vector of length 0 for no offset.
* `subset` - column indexes for `x` that should be included in the fit. Defaults to all columns.
* `convTol` - convergence criterion for relative change in deviance. Defaults to 1.0e-8.

** Details **

An intercept is not included by default. If you want one, make sure the first column of your design matrix
is all ones. Because of how `GLMNet` handles intercepts and how `LReg` interfaces with it,
this must be the first column.
If you do that, don't forget to add the column in the same place to `newx` when you call predict.

`subset` is useful if you want to include only some columns of the design matrix but you want to call `predict`
with a `newx` matrix with the same number of columns as `x`. The coefficients corresponding to columns of `x` which
are not used in the fit are set to zero. If you would like to call `predict` with a `newx` matrix that includes
only the columns that you fit on, you should subset `x` yourself before calling `lreg`.
"""
function lreg(x, y; wts=ones(y), offset=similar(y,0), subset=1:size(x,2), convTol=1.0e-8)
    if size(x, 2) == 1
        x = reshape(x, size(x, 1), 1)
    end
    subset=sort(subset)
    fitwithoffset = length(offset) > 0

    if subset == 1:size(x,2) || collect(1:size(x,2)) == subset
        subset = 1:size(x,2)
#         return LR(coef(fit(GeneralizedLinearModel, x, y, Binomial(); wts=wts, offset=offset, convTol=convTol)), subset, fitwithoffset)
        return LR(myfit(x, y, wts=wts, offset=offset, convTol=convTol), subset, fitwithoffset)
    else
        tempx = x[:, subset]
        β = zeros(eltype(x), size(x, 2))
#        β[subset] = coef(fit(GeneralizedLinearModel, tempx, y, Binomial(); wts=wts, offset=offset, convTol=convTol))
        β[subset] = myfit(tempx, y, wts=wts, offset=offset, convTol=convTol)
        return LR(β, subset, fitwithoffset)
    end
end

loss(y, xb) =
    y == one(y)? log1pexp(-xb) :
    y == zero(y)? log1pexp(xb) :
    y * log1pexp(-xb) + (one(y)-y) * log1pexp(xb)

function myfit(x, y; wts=ones(y), offset=similar(y,0), convTol=1.0e-8)
    offsets = length(offset) == 0? nothing: offset
    try
        #annoyingly, glmnet won't do intercept only  models for some reason, even with intercept = true
        # and if there is a constant column in the design matrix, that will get a coefficient of zero
        if size(x, 2) == 1 && all(x .== 1)
            offsets==nothing || error("Fitting an intercept only model with an offset is not supported")
            #intercept only
            [logit(sum(y .* wts)/sum(wts))]
        elseif all(x[:, 1] .== 1)
            #first column is intercept
            est = glmnet(x[:, 2:end], [ones(length(y)) .- y y], Binomial(), weights=wts, offsets=offsets,
                         lambda=[0.0], intercept=true, tol=convTol)
            [est.a0; est.betas[:,1]]
        else
            #no intercept
            glmnet(x, [ones(length(y)) .- y y], Binomial(), weights=wts, offsets=offsets,
                   lambda=[0.0], intercept=false, tol=convTol).betas[:,1]
        end
    catch err
        if isdefined(Main, :LREG_DEBUG) &&
           Main.LREG_DEBUG
            fname = tempname()
            open(fname, "w") do f
                serialize(f, (x, y, wts, offset, convTol))
                info("Error in lreg. x, y, wts, offset and convTol written to $fname")
            end
        end
        rethrow(err)
    end
end

end
