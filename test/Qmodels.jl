using NumericExtensions
using TargetedLearning: LReg, Qmodels, Parameters

import TargetedLearning.Qmodels: Fluctuation, computefluc

facts("Testing Qmodels") do
    
    srand(412)
    n,p = 100, 10

    w = [ones(n) rand(n,p)]
    a = round(rand(n))
    y = round(rand(n))

    qfit = lreg([w a], y)
    logitQnA1 = linpred(qfit, [w ones(n)])
    logitQnA0 = linpred(qfit, [w zeros(n)])
    logitQnAA = ifelse(a.==1, logitQnA1, logitQnA0)

    q = Qmodel(logitQnA1, logitQnA0)
    
    context("predict on unfluctuated Qmodel") do 
        @fact predict(q, a) --> logistic(ifelse(a.==1, logitQnA1, logitQnA0))
    end
    
    gn1 = predict(lreg(w,a), w)
    gna = ifelse(a.==1, gn1, 1-gn1)
    hAA = (2a-1)./gna
    hA1 = 1./gn1
    fluc = lreg(hAA, y, offset=logitQnAA)
    param=ATE()
    theirfluc = computefluc(q, param, gn1, a, y)
    fluctuate!(q, param, gn1, a, y)
        
    context("fluctuation") do
        @fact theirfluc.epsilon.β --> roughly(fluc.β)
        @fact q.flucseq[1].epsilon.β --> roughly(fluc.β)
    end

    context("predict on fluctated Qmodel") do
        @fact predict(q, ones(n)) --> roughly(logistic(logitQnA1 + hA1 * fluc.β[1]))
    end
    
    context("defluctuate!") do
        @fact defluctuate!(q) --> is_a(Fluctuation{Float64})
        @fact length(q.flucseq) --> 0
    end
end

