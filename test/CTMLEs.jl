using TargetedLearning: LReg, CTMLEs, Parameters
import TargetedLearning.CTMLEs: CTMLE, Qmodel, Qchunk, FluctuationInfo

facts("Test CTMLEs") do
    srand(1234)
    n,p = 100, 5
    w = [ones(n) rand(n, p)]
    a = round(rand(n))
    y = round(rand(n))

    qfit = lreg([w a], y)
    logitQnA1 = linpred(qfit, [w ones(n)])
    logitQnA0 = linpred(qfit, [w zeros(n)])

    context("subsetting regimens") do
        @fact CTMLEs.subset(StaticRegimen(1.0), [1,2,3]) => StaticRegimen(1.0)
        @fact CTMLEs.subset(DynamicRegimen([1.0, 0.0, 1.0]), [1, 3]) =>
            DynamicRegimen([1.0, 1.0])
        subset_ate = CTMLEs.subset(ATE(StaticRegimen(1.0), DynamicRegimen([1.0, 0.0, 1.0]), ),  [1, 3])
        @fact subset_ate.d1 => StaticRegimen(1)
        @fact subset_ate.d0 => DynamicRegimen([1.0, 1.0])
    end

    context("making chunk subsets") do
        qc = CTMLEs.make_chunk_subset(logitQnA1, logitQnA0, w, a, y, ATE(), 1:3, [0.0, 1.0])
        @fact qc => is_a(Qchunk)
        @fact CTMLEs.nobs(qc) => 3
        @fact qc.risk => CTMLEs.risk(Qmodel(logitQnA1[1:3], logitQnA0[1:3]), a[1:3], y[1:3])
    end

    context("FluctuationInfo") do
        g1 = LReg.LR(zeros(1), [1], false)
        g2 = LReg.LR(zeros(2), [1, 3], false)
        g1b = LReg.LR(zeros(2), [1, 5], false)

        f1 = FluctuationInfo(1, [g1], [true])
        @fact f1.covar_order => Vector{Int}[[1]]
        @fact f1.added_each_fluc => Vector{Int}[[1]]

        f2 = FluctuationInfo(2, [g1, g2], [true, false])
        @fact f2.covar_order =>  Vector{Int}[[1], [3]]
        @fact f2.added_each_fluc => Vector{Int}[[1, 3]]

        f3 = FluctuationInfo(2, [g1b, g2], [true, true])
        @fact f3.covar_order =>  Vector{Int}[[1, 5], [3]]
        @fact f3.added_each_fluc =>  Vector{Int}[[1, 5], [3]]
    end

    context("pcor") do
        #test pcor
        srand(235)
        xx = rand(100, 3)
        yy = rand(100, 2)
        zz = rand(100)
        oz = [ones(100) zz]

        @fact CTMLEs.pcor(xx,yy,zz) => roughly(cor(xx .- oz*(oz\xx), yy .- oz*(oz\yy)))
    end

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
