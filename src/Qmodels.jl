#Qmodel keeps track of initial Q estimate (by logistic reg) and
#sequence of g estimates and fluctuations

module Qmodels

using ..Common, ..LReg, NumericExtensions, NumericFuns

import ..LReg: linpred, predict
import StatsBase.nobs

export Qmodel, fluctuate!, defluctuate!, predict, linpred, nobs

type Fluctuation{T<:FloatingPoint}
    hA1::Vector{T}
    hA0::Vector{T}
    epsilon::LR{T}
    function Fluctuation(hA1, hA0, epsilon)
        length(hA1) == length(hA0) || error(ArgumentError("lengths of hA1 and hA0 do not match"))
        new(hA1, hA0, epsilon)
    end
end

Fluctuation{T<:FloatingPoint}(hA1::Vector{T}, hA0::Vector{T},
                              epsilon::LR{T}) = Fluctuation{T}(hA1, hA0, epsilon)    

nobs(fluc::Fluctuation) = length(fluc.hA1)

function linpred{T<:FloatingPoint}(fluc::Fluctuation{T}, a::Vector{T}; offset::Vector{T}=zeros(T, 0))
    hAA = ifelse(a.==1, fluc.hA1, fluc.hA0)
    linpred(fluc.epsilon, hAA, offset=offset)
end

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

# Qmodel{T<:FloatingPoint}(logitQnA1::Vector{T}, logitQnA0::Vector{T}, param::Parameter{T})=
#     Qmodel{T}(logitQnA1, logitQnA0, param, Vector{T}[], Vector{LR{Float64}}[])

function linpred{T<:FloatingPoint}(q::Qmodel{T}, a::Vector{T})
    r = ifelse(a .==1, q.logitQnA1, q.logitQnA0)
    for fluc in q.flucseq
        r = linpred(fluc, a, offset=r)
    end
    r
end

predict{T<:FloatingPoint}(q::Qmodel{T}, a::Vector{T}) = map1!(LogisticFun(), linpred(q,a))

function computefluc(q::Qmodel, param::Parameter, gn1, a, y)
    #computes fluctuation for a given q and g
    offset = linpred(q, a)
    hA1 = fluccovar(param, ones(nobs(q)), gn1)
    hA0 = fluccovar(param, zeros(nobs(q)), gn1)
    hAA = ifelse(a.==1, hA1, hA0)
    epsilon = lreg(hAA, y, offset=offset)
    Fluctuation(hA1, hA0, epsilon)
end

function fluctuate!{T<:FloatingPoint}(q::Qmodel{T}, fluc::Fluctuation{T})
    nobs(q) == nobs(fluc) || error(ArgumentError("nobs of q and fluc do not match"))
    push!(q.flucseq, fluc)
    q
end

function fluctuate!{T<:FloatingPoint}(q::Qmodel{T}, param::Parameter{T}, gn1::Vector{T}, a::Vector{T}, y::Vector{T})
    fluctuate!(q, computefluc(q, param, gn1, a, y))
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

# finalg(q::Qmodel) = q.gseq[end]

end
