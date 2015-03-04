module TestQmodels

using Base.Test, NumericExtensions

using TargetedLearning
using TargetedLearning: LReg, Qmodels, Parameters

import TargetedLearning.Qmodels.Fluctuation

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

@test_approx_eq logistic(ifelse(a.==1, logitQnA1, logitQnA0)) predict(q, a)

gn1 = predict(lreg(w,a), w)

gna = ifelse(a.==1, gn1, 1-gn1)
hAA = (2a-1)./gna
hA1 = 1./gn1
fluc = lreg(hAA, y, offset=logitQnAA)

param=ATE()

theirfluc = computefluc(q, param, gn1, a, y)

@test_approx_eq fluc.β theirfluc.epsilon.β

fluctuate!(q, param, gn1, a, y)

@test_approx_eq fluc.β q.flucseq[1].epsilon.β
@test_approx_eq logistic(logitQnA1 + hA1 * fluc.β[1]) predict(q, ones(n))

fluc = defluctuate!(q)
@test isa(fluc, Fluctuation{Float64})
@test length(q.flucseq) == 0

end
