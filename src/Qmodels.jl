#Qmodel keeps track of initial Q estimate (by logistic reg) and
#sequence of g estimates and fluctuations

module Qmodels

using ..Parameters, ..LReg, NumericExtensions, NumericFuns

import ..LReg: linpred, predict

export Qmodel, computefluc, fluctuate!, defluctuate!, predict, linpred, psi_ic

type Qmodel{T<:FloatingPoint}
    logitQnA1::Vector{T} # \logit\bar{Q}_{n}(A=1, W)
    logitQnA0::Vector{T} # \logit\bar{Q}_{n}(A=0, W)
    param::Parameter{T} # parameter mapping
    gseq::Vector{Vector{T}} #gseq[i] = ith vector of g_{n}(A=1|W)
    flucseq::Vector{LR{T}} #flucseq[i] = ith fluctuation LR
end

Qmodel{T<:FloatingPoint}(logitQnA1::Vector{T}, logitQnA0::Vector{T}, param::Parameter{T})=
    Qmodel{T}(logitQnA1, logitQnA0, param, Vector{T}[], Vector{LR{Float64}}[])

function linpred{T<:FloatingPoint}(q::Qmodel{T}, a::Vector{T})
    r = ifelse(a .==1, q.logitQnA1, q.logitQnA0)
    for (gn1, fluc) in zip(q.gseq, q.flucseq)
        h = fluccovar(q.param, a, gn1)
        r = linpred(fluc, h, offset=r)
    end
    r
end

predict{T<:FloatingPoint}(q::Qmodel{T}, a::Vector{T}) = map1!(LogisticFun(), linpred(q,a))

function computefluc(q::Qmodel, gn1, a, y)
    #computes fluctuation for a given q and g
    offset = linpred(q, a)
    h = fluccovar(q.param, a, gn1)
    lreg(h, y, offset=offset)
end

function fluctuate!{T<:FloatingPoint}(q::Qmodel{T}, gn1::Vector{T}, fluc::LR{T})
    #adds g and fluc to q
    @assert length(q.gseq) == length(q.flucseq)
    push!(q.gseq, gn1)
    push!(q.flucseq, fluc)
    q
end

function fluctuate!{T<:FloatingPoint}(q::Qmodel{T}, gn1::Vector{T}, a::Vector{T}, y::Vector{T})
    fluctuate!(q, gn1, computefluc(q, gn1, a, y))
end

function defluctuate!(q::Qmodel)
    #removes last fluctuation and g
    @assert length(q.gseq) == length(q.flucseq) > 0
    (pop!(q.gseq), pop!(q.flucseq))
end

function psi_ic{T<:FloatingPoint}(param::Mean{T}, q::Qmodel{T}, gn1::Vector{T}, A, Y)
    n = length(A)
    dv = dvec(param.d, n)
    h = fluccovar(param, A, gn1)
    QnAd = predict(q, dv)
    psi = mean(QnAd)
    ic = h .* (Y .- QnAd)
    (psi, ic)
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
