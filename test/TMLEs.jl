module TestTMLEs

using Base.Test
using TargetedLearning: TMLEs, LReg, Parameters

n,p = 100, 10
w = [ones(n) rand(n, p)]
a = round(rand(n))
y = round(rand(n))

qfit = lreg([w a], y)
logitQnA1 = linpred(qfit, [w ones(n)])
logitQnA0 = linpred(qfit, [w zeros(n)])
gn1 = predict(lreg(w,a),w)
est=tmle(logitQnA1, logitQnA0, gn1, a, y, param=ATE(1,0))

@test isa(est, TMLE)

hA1 = 1./gn1
hA0 = -1./(1 - gn1)
hAA = ifelse(a.==1, hA1, hA0)
fluc = lreg(hAA, y, offset = ifelse(a.==1, logitQnA1, logitQnA0))
QstarA1 = predict(fluc, hA1, offset=logitQnA1)
QstarA0 = predict(fluc, hA0, offset=logitQnA0)
psi = mean(QstarA1 .- QstarA0)
ic = hAA .* (y .- ifelse(a.==1, QstarA1, QstarA0)) .+ QstarA1 .- QstarA0 .- psi

@test_approx_eq est.psi psi
@test_approx_eq est.ic ic

end
