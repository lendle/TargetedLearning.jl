module TestLReg

using Base.Test

using TargetedLearning.LReg
using NumericExtensions

srand(10)
x = rand(50, 3)
y = round(rand(50))
newx = rand(20, 3)

lrfit = lreg(x, y)
@test isa(lrfit, LReg.LR)
@test_approx_eq predict(lrfit, newx) logistic(newx * lrfit.β)

off = rand(20)
@test_approx_eq predict(lrfit, newx, offset=off) logistic(newx * lrfit.β .+ off)

lrfitwts = lreg(x, y, wts=ones(50))
@test_approx_eq lrfitwts.β lrfit.β

rwts = rand(50)
lrfitwts2 = lreg(x, y, wts=rwts)
@test maxabsdiff(lrfit.β, lrfitwts2.β) > 0.1

sp1 = lreg(x, y, subset=1:3)
@test_approx_eq lrfit.β sp1.β


sp2 = lreg(x, y, subset=[3,1])
@test_approx_eq sp2.β[[1,3]] LReg.lreg(x[:, [1,3]], y).β
@test_approx_eq sp2.β[2] 0.0
@test sp2.idx == [1,3]

@test_approx_eq sp2.beta LReg.sslreg(x, y, [1,3]).beta

end
