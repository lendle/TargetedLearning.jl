module TestCTMLEs
using TargetedLearning: CTMLEs, CTMLEs.Strategies
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

@test_throws CTMLEOpts()

@test isa(CTMLEOpts(searchstrategy=ForwardStepwise(),
                    QWidx = [1,2],
                    allgWidx = IntSet(1:3),
                    cvscheme = StratifiedKfold,
                    cvschemeopts = 3
                    ), CTMLEOpts)

#Qmodel

import TargetedLearning.CTMLEs: Qmodel, predict!, predict, computefluc, fluctuate!, fluctuate, defluctuate!, risk
using LReg

n,p = 100, 10

w = [ones(n) rand(n, p)]
a = round(rand(n))
y = round(rand(n))

qlreg = lreg(w,y)
q = Qmodel(qlreg)

end

reload(joinpath(Pkg.dir("TargetedLearning"), "src", "TargetedLearning.jl"))
using TargetedLearning.CTMLEs
using MLBase
using Base.Test


n,p = 100, 10

w = [ones(n) rand(n,p)]
a = round(rand(n))
y = round(rand(n))

#=
QWAidx = IntSet(1,2,3, 11)

train_idx = [1:90]

val_idx = setdiff(1:n, train_idx)

train_dat = (w[train_idx, :], a[train_idx], y[train_idx])
val_dat = (w[val_idx, :], a[val_idx], y[val_idx])

#fit initial Q and build fluctuations on training data
train_qfit =TargetedLearning.CTMLEs.Qmodel(TargetedLearning.LReg.sparselreg([train_dat[1] train_dat[2]], train_dat[3], QWAidx))

train_qfit.gseq
3


opts=TargetedLearning.CTMLEs.CTMLEOpts(cvscheme=StratifiedRandomSub, cvschemeopts=(90, 1), QWidx = [1,2,3])

train_risk, val_risk = TargetedLearning.CTMLEs.build_Q!(train_qfit, train_dat, val_dat, opts=opts)

qfit = TargetedLearning.CTMLEs.Qmodel(TargetedLearning.LReg.sparselreg([w a], y, QWAidx));3


TargetedLearning.CTMLEs.build_Q!(qfit, (w, a, y), k = 1, opts=opts)

qfit


TargetedLearning.CTMLEs.ctmle(w,a,y)

=#

@test isa(ctmle(w,a,y), CTMLE)

@test isa(ctmle(w,a,y, QWidx=[1]), CTMLE)
@test isa(ctmle(w,a,y, QWidx=[1,2,3]), CTMLE)
@test isa(ctmle(w,a,y, allgWidx=[1,3,4]), CTMLE)
@test isa(ctmle(w,a,y, cvscheme=StratifiedRandomSub, cvschemeopts=(90, 3)), CTMLE)
@test isa(ctmle(w,a,y, cvscheme=StratifiedKfold, cvschemeopts=(10,)), CTMLE)

TargetedLearning.CTMLEs.ctmle(w,a,y, searchstrategy=TargetedLearning.CTMLEs.Strategies.PreOrdered(TargetedLearning.CTMLEs.Strategies.LogisticOrdering(),3))
