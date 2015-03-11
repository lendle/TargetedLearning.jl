#Qmodel keeps track of initial Q estimate (by logistic reg) and
#sequence of g estimates and fluctuations

module Qmodels

using Docile
@document

using ..Common, ..LReg, NumericExtensions, NumericFuns

import ..LReg: linpred, predict
import StatsBase.nobs

export Qmodel, fluctuate!, defluctuate!, predict, linpred, nobs, weightedcovar, finalfluc

"""
A `Fluctuation` holds fluctuation covariates and weights. The weight times the fluctuation covariate is the so called "clever covariate" in the targeted
learning literature.
"""
type Fluctuation{T<:FloatingPoint}
    hA1::Vector{T}
    hA0::Vector{T}
    wts::Vector{T}
    epsilon::LR{T}
    function Fluctuation(hA1, hA0, wts, epsilon)
        length(hA1) == length(hA0) == length(wts) || error(ArgumentError("lengths of hA1, hA0 and wts do not match"))
        new(hA1, hA0, wts, epsilon)
    end
end

Fluctuation{T<:FloatingPoint}(hA1::Vector{T}, hA0::Vector{T}, wts::Vector{T},
                              epsilon::LR{T}) = Fluctuation{T}(hA1, hA0, wts, epsilon)

nobs(fluc::Fluctuation) = length(fluc.hA1)

"""Computes the weight times flucutation covariate, sometimes called the "clever covariate"

** Arguments **

* `fluc` - fluctuation
* `a` - treatment vector
"""
function weightedcovar{T<:FloatingPoint}(fluc::Fluctuation{T}, a::Vector{T})
    ifelse(a.==1, fluc.hA1, fluc.hA0) .* fluc.wts
end

"""
Computes the fluctuated linear predictor given an offset

** Arguments ** 

* `fluc` - fluctuation
* `a` - treatment vector

** Keyword Arguments **

* `offset` - offset term, defaults to zeros, but the default should generally not be used
"""
function linpred{T<:FloatingPoint}(fluc::Fluctuation{T}, a::Vector{T}; offset::Vector{T}=zeros(T, 0))
    hAA = ifelse(a.==1, fluc.hA1, fluc.hA0)
    linpred(fluc.epsilon, hAA, offset=offset)
end

"""
The `Qmodel` type represents an estimate of \(\bar{Q}(a,w) = E(Y\mid A=a, W=w)\).

Initial estimate and fluctuations.
"""
type Qmodel{T<:FloatingPoint}
    logitQnA1::Vector{T} # \logit\bar{Q}_{n}(A=1, W)
    logitQnA0::Vector{T} # \logit\bar{Q}_{n}(A=0, W)
    flucseq::Vector{Fluctuation{T}} #flucseq[i] = ith fluctuation LR
    function Qmodel(logitQnA1, logitQnA0)
        length(logitQnA1) == length(logitQnA0) || error(ArgumentError("lengths of logitQnA1 and logitQnA0 do not match"))
        new(logitQnA1, logitQnA0, Fluctuation{T}[])
    end
end

Qmodel{T<:FloatingPoint}(logitQnA1::Vector{T}, logitQnA0::Vector{T}) = Qmodel{T}(logitQnA1, logitQnA0)

nobs(q::Qmodel) = length(q.logitQnA1)

"Returns the last fluctuation of `q`"
function finalfluc(q::Qmodel)
    length(q.flucseq) > 0 || error("q has not been fluctuated")
    q.flucseq[end]
end

# Qmodel{T<:FloatingPoint}(logitQnA1::Vector{T}, logitQnA0::Vector{T}, param::Parameter{T})=
#     Qmodel{T}(logitQnA1, logitQnA0, param, Vector{T}[], Vector{LR{Float64}}[])
"""Computes the linear predictor of an estimated `Qmodel` given `a`

** Arguments **

* `q` - `Qmodel`
* `a` - vector of treatments

"""
function linpred{T<:FloatingPoint}(q::Qmodel{T}, a::Vector{T})
    r = ifelse(a .==1, q.logitQnA1, q.logitQnA0)
    for fluc in q.flucseq
        r = linpred(fluc, a, offset=r)
    end
    r
end

"""Computes the prediction of an estimated `Qmodel` given `a`

** Arguments **

* `q` - `Qmodel`
* `a` - vector of treatments

"""
predict{T<:FloatingPoint}(q::Qmodel{T}, a::Vector{T}) = map1!(LogisticFun(), linpred(q,a))

"""
Computes a new fluctuation for a (possible already fluctuated) initial `Qmodel`

** Arguments **

* `q` - `Qmodel`
* `param` - target parameter
* `gn1` - estimated propensity score
* `A` - observed treatments
* `Y` - observed outcomes

** Keyword Arguments ** 

* `method` - A symbol, either `:unweighted` (the default) or `:weighted`.
If `:unweighted`, the "standard" fluctuation is performed, there the fluctuation covariate is divided
by \(g_n(A_i\mid W_i)\), and weights are set to 1. 
If `:weighted`, the fluctuation covariate is used directly, and weights are set to the reciprical
of \(g_n(A_i\mid W_i)\) where \(g_n(A_i \mid W_i)\) is computed as `gn1[i]` if `A[i]` is 1, or `1-gn1[i]` otherwise.
"""
function computefluc(q::Qmodel, param::Parameter, gn1, A, Y; method::Symbol=:unweighted)
    #computes fluctuation for a given q and g
    offset = linpred(q, A)
    if method == :unweighted
        wts = ones(nobs(q))
        hA1 = fluccovar(param, ones(nobs(q))) ./ gn1
        hA0 = fluccovar(param, zeros(nobs(q))) ./ (1 .- gn1)
    elseif method == :weighted
        gna = ifeflse(A.==1, gn1, (1 .- gn1))
        wts = 1 ./ gna
        hA1 = fluccovar(param, ones(nobs(q)))
        hA0 = fluccovar(param, zeros(nobs(q)))
    else
        error(ArgumentError("method $method not supported"))
    end   
    hAA = ifelse(A.==1, hA1, hA0)
    epsilon = lreg(hAA, Y, offset=offset, wts=wts)
    Fluctuation(hA1, hA0, wts, epsilon)
end

function fluctuate!{T<:FloatingPoint}(q::Qmodel{T}, fluc::Fluctuation{T})
    nobs(q) == nobs(fluc) || error(ArgumentError("nobs of q and fluc do not match"))
    push!(q.flucseq, fluc)
    q
end

"""
Computes and adds a new fluctuation to a `Qmodel`


** Arguments **

* `q` - `Qmodel`
* `param` - target parameter
* `gn1` - estimated propensity score
* `A` - observed treatments
* `Y` - observed outcomes

** Keyword Arguments ** 

* `method` - A symbol, either `:unweighted` (the default) or `:weighted`.
See documentation for `computefluc` for details.
"""
function fluctuate!{T<:FloatingPoint}(q::Qmodel{T}, param::Parameter{T}, gn1::Vector{T},
                                      A::Vector{T}, Y::Vector{T};
                                      method::Symbol=:unweighted)
    fluctuate!(q, computefluc(q, param, gn1, A, Y))
end

function defluctuate!(q::Qmodel)
    pop!(q.flucseq)
end

# function risk(q::Qmodel, w, a, y; pen=true)
#     #calculate penalized log likelihood as RSS + tau(=1 here?) * var(ic)
#     # need to implement log likelihood
#     # note: not scaled by n. ask Susan.

#     loss = sum(LReg.Loss(), y, predict(q, w, a, :link))

#     if pen
#         n = length(y)
#         h = predict(finalg(q), w)
#         map1!(Gatoh(), h, a)

#         QnA1 = predict(q, w, ones(n), :prob)
#         QnA0 = predict(q, w, zeros(n), :prob)
#         QnAA = ifelse(a.==1, QnA1, QnA0)

#         resid = y .- QnAA

#         loss += var(h .* resid .+ QnA1 .- QnA0)
#     end
#     loss
# end


end
