using TargetedLearning.LReg
using StatsFuns

facts("Testing LReg") do

    n, p, newn = 50, 3, 20
    srand(10)
    x = rand(n, p)
    y = round(rand(n))
    newx = rand(newn, p)

    context("without offest or weights") do
        lrfit = lreg(x, y)
        @fact lrfit --> is_a(LR)
        @fact predict(lrfit, newx) --> roughly(logistic(newx * lrfit.β))

        @fact_throws predict(lrfit, newx, offset=rand(newn))

        pred = logistic(x * lrfit.β)
        total_loss = 0.0
        lp=linpred(lrfit, x)
        for i in 1:length(y) #NumericExtensions
          total_loss += LReg.loss(y[i], lp[i])
        end
        @fact total_loss/length(y) --> roughly(mean(- y .* log(pred) .- (1-y) .* log(1 .- pred)))
    end

    context("with offset") do
        off = rand(n)
        newoff = rand(newn)
        lrfitoff = lreg(x, y, offset=off)
        @fact predict(lrfitoff, newx, offset=newoff) -->  roughly(logistic(newx * lrfitoff.β .+ newoff))
        @fact_throws predict(lrfitoff, newx)
    end

    context("with weights") do
        lrfitwts = lreg(x, y, wts=ones(50))
        @fact lrfitwts.β --> roughly(lreg(x,y).β)
        rwts = rand(50)
        lrfitwts2 = lreg(x, y, wts=rwts)
        @fact lrfitwts2.β --> not(roughly(lrfitwts.β))
    end

    context("fitting on a subset of columns") do
        sslrfit = lreg(x, y, subset=[3,1])
        @fact sslrfit.β[[1,3]] --> roughly(lreg(x[:, [1,3]], y).β)
        @fact sslrfit.β[2] --> 0.0
        @fact sslrfit.idx --> [1,3]
        @fact sslrfit.β --> lreg(x, y, subset=[1,3]).β
    end

    context("works when x is a vector") do
        lrvec = lreg(rand(10), round(rand(10)))
        @fact lrvec --> is_a(LR)
        @fact predict(lrvec, rand(10)) --> is_a(Vector)
    end

    context("intercept only") do
        @pending "without weights" --> nothing
        @pending "with weights" --> nothing
    end

end
