using TargetedLearning: LReg, CTMLEs
import TargetedLearning.CTMLEs.CTMLE

facts("Test CTMLEs") do
    srand(6578)
    n,p = 100, 5
    w = [ones(n) rand(n, p)]
    a = round(rand(n))
    y = round(rand(n))

    qfit = lreg([w a], y)
    logitQnA1 = linpred(qfit, [w ones(n)])
    logitQnA0 = linpred(qfit, [w zeros(n)])
    
    context("It runs...") do
        est=ctmle(logitQnA1, logitQnA0, w, a, y)
        @fact est => is_a(CTMLE)
        est2 = ctmle(logitQnA1, logitQnA0, w, a, y, searchstrategy=CTMLEs.PreOrdered(CTMLEs.LogisticOrdering()))
        @fact est2 => is_a(CTMLE)
        est3 = ctmle(logitQnA1, logitQnA0, w, a, y, searchstrategy=CTMLEs.PreOrdered(CTMLEs.PartialCorrOrdering()))
        @fact est3 => is_a(CTMLE)
    end
    @pending "Actually test that it works" => nothing
end
