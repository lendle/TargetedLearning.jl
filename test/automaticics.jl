module TestICs
using Base.Test

using TargetedLearning.Common

#simulate scalar estimates for EY1, EY0
srand(1234)
n = 50
a = rand(n)
b = rand(n)
ey1 = ScalarEstimate(mean(a), a .- mean(a), "EY1")
ey0 = ScalarEstimate(mean(b), b .- mean(b), "EY0")

ate = name!(ey1 - ey0, "ATE")
@test_approx_eq ate.ic  ey1.ic .- ey0.ic

#see http://onlinelibrary.wiley.com/doi/10.1002/sim.3445/abstract;jsessionid=F6B735D18167EB40C4EEB55440235B71.f04t01 appendix
# for influence curves for logRR and logOR
logrr = log(ey1/ey0)
@test_approx_eq logrr.ic (1/ey1.psi) .* ey1.ic .- (1/ey0.psi) .* ey0.ic

logor = log((ey1/(1-ey1))/(ey0/(1-ey0)))
@test_approx_eq logor.ic (1/ey1.psi + 1/(1-ey1.psi)) .* ey1.ic - (1/ey0.psi + 1/(1-ey0.psi)) .* ey0.ic

logor2 = log(ey1) - log(1-ey1) - log(ey0) + log(1-ey0)
@test_approx_eq logor.ic logor2.ic

end
