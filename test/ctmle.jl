module TestCTMLEs
using TargetedLearning: CTMLEs, CTMLEs.Strategies, LReg
using Base.Test

#test pcor
import TargetedLearning.CTMLEs.pcor
srand(235)
x = rand(100, 3)
y = rand(100, 2)
z = rand(100)
oz = [ones(100) z]

@test_approx_eq pcor(x,y,z) cor(x .- oz*(oz\x), y .- oz*(oz\y))

#construction of CTMLEOpts

import TargetedLearning.CTMLEs.CTMLEOpts

@test_throws ErrorException CTMLEOpts()

@test isa(CTMLEOpts(searchstrategy=ForwardStepwise(),
                    QWidx = [1,2],
                    allgWidx = IntSet(1:3),
                    cvscheme = StratifiedKfold,
                    cvschemeopts = 3
                    ), CTMLEOpts)

#Qmodel

import TargetedLearning.CTMLEs: Qmodel, predict!, predict, computefluc, fluctuate!, fluctuate, defluctuate!, risk

n,p = 100, 10

w = [ones(n) rand(n, p)]
a = round(rand(n))
y = round(rand(n))

qlreg = LReg.SSLR(lreg(w,y).beta, IntSet(1:p))



q = Qmodel(qlreg)

#need to test fluctuate, defluctuate, etc. here.

###################


import MLBase: StratifiedKfold, StratifiedRandomSub

n,p = 100, 10

w = [ones(n) rand(n,p)]
a = round(rand(n))
y = round(rand(n))

@test isa(ctmle(w,a,y), CTMLE)

@test isa(ctmle(w,a,y, QWidx=[1]), CTMLE)
@test isa(ctmle(w,a,y, QWidx=[1,2,3]), CTMLE)
@test isa(ctmle(w,a,y, allgWidx=[1,3,4]), CTMLE)
@test isa(ctmle(w,a,y, cvscheme=StratifiedRandomSub, cvschemeopts=(90, 3)), CTMLE)
@test isa(ctmle(w,a,y, cvscheme=StratifiedKfold, cvschemeopts=(10,)), CTMLE)

@test isa(ctmle(w,a,y, searchstrategy=PreOrdered(LogisticOrdering(),3)), CTMLE)

end
