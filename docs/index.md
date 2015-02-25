# About

Targeted minimum loss-based estimation (TMLE) (sometimes targeted maximum likelihood estimation) is a general framework for constructing regular, asymptotically linear estimators for pathwise differentiable parameters with additional properties such as [asymptotic efficiency](https://en.wikipedia.org/wiki/Efficiency_%28statistics%29#Asymptotic_efficiency) and double robustness. For background and details, see [Targeted Learning](https://www.springer.com/statistics/statistical+theory+and+methods/book/978-1-4419-9781-4) by van der Laan and Rose, or [articles](http://scholar.google.com/scholar?q=targeted+estimation+tmle) on TMLE.

TargetedLearing.jl is a [Julia](julialang.org) package that implements a TMLE and collaborative TMLE (CTMLE) to estimate a handful of statistical parameters. In particular, we can currently estimate the (statistical parameter corresponding to the) [counterfactual](https://en.wikipedia.org/wiki/Rubin_causal_model) mean of some outcome under a static or dynamic single time point treatment with baseline covariates or the difference in counterfactual means. Additionally, arbitrary transformations of these parameters can be estimated, and inference is performed [automatically](user-guide/influencecurves.md).  With a little massaging, the functionality of this package can also be used to estimate the mean of an outcome subject to missingness.

## Contents

* If you're new to Julia, go [here](user-guide/julia.md) to learn why this package is implemented in Julia and enough about it to get you started.
* For details on the estimation problem and causal assumptions, go [here](user-guide/estimation.md).
* For an overview about how the CTMLE procedure works, go [here](user-guide/ctmle.md).
* For information on how we perform inference on arbitrary transformations of estimates, go [here](user-guide/influencecurves.md).
* API docs are [here](asdf).
* To obtain a special [dialing wand](https://www.youtube.com/watch?v=OqjF7HKSaaI) please mash the keypad with your palm now.

# Bugs? Questions?

Chances are someone else has the same issue, so don't email me. Instead, look at the [issues page](https://github.com/lendle/TargetedLearning.jl/issues) to see if you can find anyone else talking about it, or open a new issue.

# Want to contribute?

There's a link to the [github repository](https://github.com/lendle/TargetedLearning.jl/) in the top right of every page of the documentation. Pull requests welcome!

