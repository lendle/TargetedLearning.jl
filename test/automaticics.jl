using TargetedLearning.Common

facts("Testing automatic influence curves") do
    #simulate scalar estimates for EY1, EY0
    srand(1234)
    n = 50
    a = rand(n)
    b = rand(n)
    ey1 = ScalarEstimate(mean(a), a .- mean(a), "EY1")
    ey0 = ScalarEstimate(mean(b), b .- mean(b), "EY0")
    @fact (ey1 - ey0).ic --> roughly(ey1.ic .- ey0.ic)
    
    #see http://onlinelibrary.wiley.com/doi/10.1002/sim.3445/abstract appendix
    # for influence curves for logRR and logOR
    @fact log(ey1/ey0).ic --> roughly((1/ey1.psi) .* ey1.ic .- (1/ey0.psi) .* ey0.ic)
    
    logor = log((ey1/(1-ey1))/(ey0/(1-ey0)))
    @fact logor.ic --> roughly((1/ey1.psi + 1/(1-ey1.psi)) .* ey1.ic - (1/ey0.psi + 1/(1-ey0.psi)) .* ey0.ic)

    logor2 = log(ey1) - log(1-ey1) - log(ey0) + log(1-ey0)
    @fact logor.ic --> roughly(logor2.ic)
    
    ey1tx1 = ey1 * ey1 * ey1 / sqrt(ey1)
    ey1tx2 = ey1 ^ 2.5
    @fact ey1tx1.psi --> roughly(ey1tx2.psi)
    @fact ey1tx1.ic --> roughly(ey1tx2.ic)
end

