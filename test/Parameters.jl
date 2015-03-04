using TargetedLearning: Parameters

facts("Testing Parameters") do
    context("Regimen") do
        @fact regimen(1) => StaticRegimen(1.0)
        d = regimen([1, 0, 1])
        @fact d.a => [1.0, 0.0, 1.0]
        @fact d === regimen(d) => true
    end
    
    srand(412)
    n,p = 100, 10
    w = [ones(n) rand(n,p)]
    a = round(rand(n))
    y = round(rand(n))
    gn1 = predict(lreg(w,a), w)
    
    context("fluccovar") do
        ey1 = Mean(1.0)
        @fact fluccovar(ey1, a, gn1) => roughly(a./gn1)
        @fact fluccovar(ey1, ones(n), gn1) => roughly(1./gn1)
        
        ate = ATE(ones(n), 0.0)
        gna = ifelse(a .== 1, gn1, 1-gn1)
        @fact fluccovar(ate, a, gn1) => roughly((2a-1) ./ (gna))
        @fact fluccovar(ate, ones(n), gn1) => roughly(1 ./ gn1)
        @fact fluccovar(ate, zeros(n), gn1) => roughly(-1 ./ (1-gn1))
        
        d = [ones(div(n,2)), zeros(div(n,2))]
        eyd = Mean(d)
        gnd = ifelse(d.==1, gn1, 1-gn1)
        hAA = ifelse(a.==d, 1./gnd, 0.0)
        @fact fluccovar(eyd, a, gn1) => roughly(hAA)
        hA1 = ifelse(1.==d, 1./gnd, 0.0)
        @fact fluccovar(eyd, ones(n), gn1) => roughly(hA1)
        hA0 = ifelse(0.==d, 1./gnd, 0.0)
        @fact fluccovar(eyd, zeros(n), gn1) => roughly(hA0)
    end
end
