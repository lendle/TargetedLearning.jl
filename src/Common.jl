module Common

export ScalarEstimate, coef, vcov, nobs, stderr, confint, coeftable

using Distributions

import Base.var
#borrowing some names from StatsBase, but not the StatsModels type
import StatsBase: coef, vcov, nobs, stderr, confint, coeftable, CoefTable

abstract Estimate

stderr(est::Estimate) = sqrt(diag(vcov(est)))
function confint(est::Estimate, level=0.95)
    [coef(est) coef(est)] .+ [1. -1.] .* quantile(Normal(),(1. - level)/2.) .* stderr(est)
end

abstract ScalarEstimate <: Estimate

coef(est::ScalarEstimate) = est.psi
nobs(est::ScalarEstimate) = est.n
vcov(est::ScalarEstimate) = reshape([var(est.ic)/nobs(est)], 1,1)
paramnames(est::ScalarEstimate) = [est.estimand]

function coeftable(est::ScalarEstimate)
    CoefTable([coef(est) stderr(est) confint(est)],
              ["Estimate", "Std. Error", "Lower 95% CL", "Upper 95% CL"],
              paramnames(est))
end
end
