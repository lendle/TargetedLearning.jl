module TestLReg

using Base.Test

import TargetedLearning.LReg

srand(10)
x = rand(50, 3)
y = round(rand(50))

lrfit = LReg.lreg(x, y)
@test isa(lrfit, LReg.LR)

splrfit = LReg.sparselreg(x, y, [1:3])
@test isa(splrfit, LReg.SSLR)
@test lrfit.beta == splrfit.beta

sslrfit = LReg.sslreg(x, y, [1:3])
@test isa(sslrfit, LReg.SSLR)
@test lrfit.beta == sslrfit.beta

sp2 = LReg.sparselreg(x, y, [1,3])
@test sp2.beta[[1,3]] == LReg.lreg(x[:, [1,3]], y).beta
@test sp2.beta[2] == 0.0

@test_approx_eq sp2.beta LReg.sslreg(x, y, [1,3]).beta

end
