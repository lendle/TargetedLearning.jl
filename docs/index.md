# About

Targeted minimum loss-based estimation (TMLE) (sometimes targeted maximum likelihood estimation) is a general framework for constructing regular, asymptotically linear estimators for pathwise differentiable parameters with additional properties such as [asymptotic efficiency](https://en.wikipedia.org/wiki/Efficiency_%28statistics%29#Asymptotic_efficiency) and double robustness. For background and details, see [Targeted Learning](https://www.springer.com/statistics/statistical+theory+and+methods/book/978-1-4419-9781-4) by van der Laan and Rose, or [articles](http://scholar.google.com/scholar?q=targeted+estimation+tmle) on TMLE.

TargetedLearing.jl is a package for [Julia](http://julialang.org) v0.3.x that implements a TMLE and collaborative TMLE (CTMLE) to estimate a handful of statistical parameters. In particular, we can currently estimate the (statistical parameter corresponding to the) [counterfactual](https://en.wikipedia.org/wiki/Rubin_causal_model) mean of some outcome under a static or dynamic single time point treatment with baseline covariates or the difference in counterfactual means. Additionally, arbitrary transformations of these parameters can be estimated, and inference is performed [automatically](user-guide/influencecurves.md).  With a little massaging, the functionality of this package can also be used to estimate the mean of an outcome subject to missingness.

### Contents

* [Julia](user-guide/julia.md) provides some links for resources for learning Julia, links to other packages, and an example with enough to get you started.
* [The estimation problem](user-guide/estimation.md) describes the statistical parameters that can be estimated with this package and corresponding causal parameters.
* [TMLE and CTMLE](user-guide/ctmle.md) describes how TMLE and CTMLE work to estimate the average treatment effect.
* [Automatic influence curves](user-guide/influencecurves.md) describes how we perform inference on arbitrary transformations of estimates.

# Installing

If Julia is already installed, just
```jlcon
julia> Pkg.clone("https://github.com/lendle/TargetedLearning.jl.git")
```

If not, [download](http://julialang.org/downloads/) the latest version of Julia v0.3.x.

If you're on OSX and use homebrew, there's a [tap](https://github.com/staticfloat/homebrew-julia#homebrew-julia) for that.

Many linux distributions can install Julia via the package manager, but may be out of date. If you wind up with v0.2.x, you're definitely out of date.

If you [build from source](https://github.com/julialang/julia#source-download-and-compilation), make sure you `git checkout release-0.3` so you don't wind up with the development version.

# Bugs? Questions?

Chances are someone else has the same issue, so don't email me. Instead, look at the [issues page](https://github.com/lendle/TargetedLearning.jl/issues) to see if you can find anyone else talking about it, or open a new issue.

# Want to contribute?

There's a link to the [github repository](https://github.com/lendle/TargetedLearning.jl/) in the top right of every page of the documentation. Pull requests welcome!

