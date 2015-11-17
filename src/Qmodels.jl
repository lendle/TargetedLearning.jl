#Qmodel keeps track of initial Q estimate (by logistic reg) and
#sequence of g estimates and fluctuations

module Qmodels

VERSION < v"0.4-" && using Docile

using ..Common, ..LReg
using StatsFuns

import ..LReg: linpred, predict
import StatsBase.nobs

export Qmodel, fluctuate!, defluctuate!, predict, linpred, nobs, weightedcovar, lastfluc, risk,
computefluc, valfluc

"""
A `Fluctuation` holds fluctuation covariates and weights. The weight times the fluctuation covariate is the so called "clever covariate" in the targeted
learning literature.
"""
type Fluctuation{T<:AbstractFloat}
    hA1::Vector{T}
    hA0::Vector{T}
    wts::Vector{T}
    epsilon::LR{T}
    weighted::Bool
    function Fluctuation(hA1, hA0, wts, epsilon, weighted)
        length(hA1) == length(hA0) == length(wts) || error(ArgumentError("lengths of hA1, hA0 and wts do not match"))
        new(hA1, hA0, wts, epsilon, weighted)
    end
end

Fluctuation{T<:AbstractFloat}(hA1::Vector{T}, hA0::Vector{T}, wts::Vector{T},
                              epsilon::LR{T}, weighted::Bool) =
    Fluctuation{T}(hA1, hA0, wts, epsilon, weighted)

nobs(fluc::Fluctuation) = length(fluc.hA1)

"""Computes the weight times flucutation covariate, sometimes called the "clever covariate"

** Arguments **

* `fluc` - fluctuation
* `a` - treatment vector
"""
function weightedcovar{T<:AbstractFloat}(fluc::Fluctuation{T}, a::Vector{T})
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
function linpred{T<:AbstractFloat}(fluc::Fluctuation{T}, a::Vector{T}; offset::Vector{T}=zeros(T, 0))
    hAA = ifelse(a.==1, fluc.hA1, fluc.hA0)
    linpred(fluc.epsilon, hAA, offset=offset)
end

"""
The `Qmodel` type represents an estimate of \(\bar{Q}(a,w) = E(Y\mid A=a, W=w)\).

Initial estimate and fluctuations.
"""
type Qmodel{T<:AbstractFloat}
    logitQnA1::Vector{T} # \logit\bar{Q}_{n}(A=1, W)
    logitQnA0::Vector{T} # \logit\bar{Q}_{n}(A=0, W)
    flucseq::Vector{Fluctuation{T}} #flucseq[i] = ith fluctuation LR
    function Qmodel(logitQnA1, logitQnA0)
        length(logitQnA1) == length(logitQnA0) || error(ArgumentError("lengths of logitQnA1 and logitQnA0 do not match"))
        new(logitQnA1, logitQnA0, Fluctuation{T}[])
    end
end

Qmodel{T<:AbstractFloat}(logitQnA1::Vector{T}, logitQnA0::Vector{T}) = Qmodel{T}(logitQnA1, logitQnA0)

nobs(q::Qmodel) = length(q.logitQnA1)

"Returns the last fluctuation of `q`"
function lastfluc(q::Qmodel)
    length(q.flucseq) > 0 || error("q has not been fluctuated")
    q.flucseq[end]
end

# Qmodel{T<:AbstractFloat}(logitQnA1::Vector{T}, logitQnA0::Vector{T}, param::Parameter{T})=
#     Qmodel{T}(logitQnA1, logitQnA0, param, Vector{T}[], Vector{LR{Float64}}[])
"""Computes the linear predictor of an estimated `Qmodel` given `a`

** Arguments **

* `q` - `Qmodel`
* `a` - vector of treatments

"""
function linpred{T<:AbstractFloat}(q::Qmodel{T}, a::Vector{T})
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
predict{T<:AbstractFloat}(q::Qmodel{T}, a::Vector{T}) = logistic(linpred(q,a))

function risk{T<:AbstractFloat}(q::Qmodel{T}, A::Vector{T}, Y::Vector{T})
  total_loss = 0.0
  length(Y) == nobs(q) || error()
  lp = linpred(q, A)
  for i in 1:nobs(q)
    total_loss += LReg.loss(Y[i], lp[i])
  end
  total_loss/nobs(q)
end

function compute_h_wts(param::Parameter, gn1, A; weighted::Bool=false)
    n = length(A)
    n == length(gn1) || error()
    if weighted
        gna = ifelse(A.==1, gn1, (1 .- gn1))
        wts = 1 ./ gna
        hA1 = fluccovar(param, ones(n))
        hA0 = fluccovar(param, zeros(n))
    else
        wts = ones(n)
        hA1 = fluccovar(param, ones(n)) ./ gn1
        hA0 = fluccovar(param, zeros(n)) ./ (1 .- gn1)
    end
    (hA1, hA0, wts)
end

"""
Computes a new fluctuation for a (possible already fluctuated) initial `Qmodel`

** Arguments **

* `q` - `Qmodel`
* `param` - target parameter
* `gn1` - estimated propensity score
* `A` - observed treatments
* `Y` - observed outcomes

** Keyword Arguments **

* `weighted` - A boolean, `false` by default
If `false`, the "standard" fluctuation is performed, there the fluctuation covariate is divided
by \(g_n(A_i\mid W_i)\), and weights are set to 1.
If `true`, the fluctuation covariate is used directly, and weights are set to the reciprical
of \(g_n(A_i\mid W_i)\) where \(g_n(A_i \mid W_i)\) is computed as `gn1[i]` if `A[i]` is 1, or `1-gn1[i]` otherwise.
"""
function computefluc(q::Qmodel, param::Parameter, gn1, A, Y; weighted::Bool=false)
    #computes fluctuation for a given q and g
    hA1, hA0, wts = compute_h_wts(param, gn1, A, weighted=weighted)
    hAA = ifelse(A.==1, hA1, hA0)
    offset = linpred(q, A)
    epsilon = lreg(hAA, Y, offset=offset, wts=wts)
    Fluctuation(hA1, hA0, wts, epsilon, weighted)
end

function valfluc(fluc::Fluctuation, param::Parameter, gn1, A; weighted::Bool=false)
    hA1, hA0, wts = compute_h_wts(param, gn1, A, weighted=fluc.weighted)
    epsilon = fluc.epsilon
    Fluctuation(hA1, hA0, wts, epsilon, weighted)
end

function fluctuate!{T<:AbstractFloat}(q::Qmodel{T}, fluc::Fluctuation{T})
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

* `weighted` - A boolean, `false` by default.
See documentation for `computefluc` for details.
"""
function fluctuate!{T<:AbstractFloat}(q::Qmodel{T}, param::Parameter{T}, gn1::Vector{T},
                                      A::Vector{T}, Y::Vector{T};
                                      weighted::Bool=false)
    fluctuate!(q, computefluc(q, param, gn1, A, Y, weighted=weighted))
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
