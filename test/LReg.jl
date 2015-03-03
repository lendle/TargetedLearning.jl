module TestLReg

using Base.Test

using TargetedLearning.LReg
using NumericExtensions

srand(10)
x = rand(50, 3)
y = round(rand(50))
newx = rand(20, 3)

lrfit = lreg(x, y)
@test isa(lrfit, LR)
@test_approx_eq predict(lrfit, newx) logistic(newx * lrfit.β)
newoff = rand(20)
@test_throws ArgumentError predict(lrfit, newx, offset=newoff)

pred = logistic(x * lrfit.β)
@test_approx_eq mean(- y .* log(pred) .- (1-y) .* log(1 .- pred)) mean(LReg.Loss(), y, linpred(lrfit, x))

lrfitoff = lreg(x, y, offset=rand(50))
@test_approx_eq predict(lrfitoff, newx, offset=newoff) logistic(newx * lrfitoff.β .+ newoff)
@test_throws ArgumentError predict(lrfitoff, newx)

lrfitwts = lreg(x, y, wts=ones(50))
@test_approx_eq lrfitwts.β lrfit.β

rwts = rand(50)
lrfitwts2 = lreg(x, y, wts=rwts)
@test maxabsdiff(lrfit.β, lrfitwts2.β) > 0.1

sslrfit = lreg(x, y, subset=[3,1])
@test_approx_eq sslrfit.β[[1,3]] lreg(x[:, [1,3]], y).β
@test_approx_eq sslrfit.β[2] 0.0
@test sslrfit.idx == [1,3]

@test_approx_eq sslrfit.β lreg(x, y, subset=[1,3]).β

#check lreg works when x is a vector
lrvec = lreg(rand(10), round(rand(10)))
@test isa(lrvec, LR)
@test isa(predict(lrvec, rand(10)), Vector)

end
