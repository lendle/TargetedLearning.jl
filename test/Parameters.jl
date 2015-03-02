module TestParameters

using Base.Test, NumericExtensions

using TargetedLearning
using TargetedLearning: LReg, Parameters

# import TargetedLearning.TMLEs: Qmodel, Mean, ATE, Parameter, Regimen, StaticRegimen, DynamicRegimen, fluccovar

srand(412)
n,p = 100, 10

w = [ones(n) rand(n,p)]
a = round(rand(n))
y = round(rand(n))

gn1 = predict(lreg(w,a), w)

ey1 = Mean(1.0)

@test_approx_eq a./gn1 fluccovar(ey1, a, gn1)
@test_approx_eq 1./gn1 fluccovar(ey1, ones(n), gn1)

ate = ATE(ones(n), 0.0)

gna = ifelse(a .== 1, gn1, 1-gn1)

@test_approx_eq (2a-1) ./ (gna) fluccovar(ate, a, gn1)
@test_approx_eq 1 ./ gn1 fluccovar(ate, ones(n), gn1)
@test_approx_eq -1 ./ (1-gn1) fluccovar(ate, zeros(n), gn1)

d = [ones(div(n,2)), zeros(div(n,2))]
eyd = Mean(d)

gnd = ifelse(d.==1, gn1, 1-gn1)
hAA = ifelse(a.==d, 1./gnd, 0.0)
@test_approx_eq hAA fluccovar(eyd, a, gn1)
hA1 = ifelse(1.==d, 1./gnd, 0.0)
@test_approx_eq hA1 fluccovar(eyd, ones(n), gn1)
hA0 = ifelse(0.==d, 1./gnd, 0.0)
@test_approx_eq hA0 fluccovar(eyd, zeros(n), gn1)

d0 = [(i%2.0)::Float64 for i in 1:n]

ate_d = ATE(d,d0)

d1=d
gnd1 = ifelse(d1 .==1, gn1, 1-gn1)
gnd0 = ifelse(d0 .==1, gn1, 1-gn1)

hAA  =  float(a.==d1)./gnd1 -  float(a .==d0)./gnd0
hAd1 =     1.0./gnd1 - float(d1.==d0)./gnd0
hAd0 = float(d0.==d1)./gnd1 -      1.0./gnd0
@test_approx_eq hAA fluccovar(ate_d, a, gn1)
@test_approx_eq hAd1 fluccovar(ate_d, d1, gn1)
@test_approx_eq hAd0 fluccovar(ate_d, d0, gn1)

end

####

