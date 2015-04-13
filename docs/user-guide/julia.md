# Why Julia

This package is implemented in Julia for mainly two reason:

* It's *easy* to write pretty fast code without having to drop into another language like Fortran or C
* It's *fun*

For a new user, it's not too hard to pick up if you're coming from a language like python or Matlab or R, which is most likely the case for people reading this.

# Getting started

There are some [links](../index.md#installing) on the front page about installing Julia if you haven't done that yet.

There are a bunch of useful resources on [julialang.org/learning/](http://julialang.org/learning/). In particular, the [Julia tutorial at SciPy 2014](https://www.youtube.com/watch?v=vWkgEddb4-A) is very thorough. This [cheat sheet](http://math.mit.edu/~stevenj/Julia-cheatsheet.pdf) and [learn Julia in Y minutes](http://learnxinyminutes.com/docs/julia/) are good quick references.  The [manual and standard library docs](http://docs.julialang.org/en/release-0.3/) are great too.

To run Julia, from a terminal type `julia`, which will start a [REPL](https://en.wikipedia.org/wiki/Read-eval-print_loop). This may seem like an ancient way of doing things, but it's actually pretty nice. It has color, tab completion, history search, and even supports tab completed Unicode symbols. For example, type `\alpha<tab>` at the prompt, and `\alpha` will get transformed to `α`.

Another choice is using IJulia notebooks via [Jupyter](http://jupyter.org/) (which, confusingly, used to be part of IPython.) Notebooks are really nice for working interactively while building up an analysis and work like IPython notebooks if you're familiar with those. Installation instructions available [here](https://github.com/JuliaLang/IJulia.jl).

If you want an IDE, there's [Juno](http://junolab.org/) which is really nice. Don't bother with Julia Studio if you come across it. It is out of date and no longer maintained.

Like R, you can access Julia's help system by typing `?name` in the REPL.  The `@edit` macro is useful if you want to see the code of the function you're calling. For example, `@edit factorial(10)` opens the method of the `factorial` function for `Int`s (the type of `10`) in an editor.

## Importing data

To get data in to Julia, the easiest way might be via a CSV file. In base Julia, there's the `readcsv` function, or more generally `readdlm` for other delimiters. Check `?readdlm` for docs.

Julia does not have built in support for missing data, but it is supported by the [DataArrays.jl](https://github.com/JuliaStats/DataArrays.jl) package which is used by [DataFrames.jl](http://dataframesjl.readthedocs.org/en/latest/).
DataFrames.jl provides the `readtable` function which can read a CSV file among other things and creates a `DataFrame` object. A `DataFrame` in Julia is sort of like a `data.frame` in R.

There's also [HDF5.jl](https://github.com/timholy/HDF5.jl) if you want to use the HDF5 format for moving data around.

## Regression and prediction

The estimation procedures in TargetedLearning.jl take as input things like estimated conditional means or probabilities. The Julia ecosystem has packages for fitting models and machine learning methods, but they are generally not as mature as those in R or python.

[GLM.jl](https://github.com/JuliaStats/GLM.jl) can be used for fitting generalized linear models and [GLMNet.jl](https://github.com/simonster/GLMNet.jl) fits glms with L1 and L2 regularization.  [Orchestra.jl](https://github.com/svs14/Orchestra.jl) fits heterogeneous ensemble estimators using cross-validation. You might be able to find other useful packages on [pkg.julialang.org](http://pkg.julialang.org).

For simple linear regression models, `linreg` is available in base Julia. For logistic regression, this package provides the `lreg` function.

For R and python users, at this point it may be easier to fit initial estimators using tools in those languages. [PyCall.jl](https://github.com/stevengj/PyCall.jl) allows you to call Python functions from Julia, so something like scikit-learn can be used. It's very mature and used in many other packages. [RCall.jl](https://github.com/JuliaStats/RCall.jl) is less mature, but it lets you do something similar with R.

## Plotting

It's always nice to be able to plot things without switching languages. Julia has a few choices. Gadfly.jl and PyPlot.jl are two good choices listed [here](http://julialang.org/downloads/plotting.html). There is also [Winston.jl](https://github.com/nolta/Winston.jl) which is not listed there. It's sort of like Matlab's plotting system, and nice for simple and quick stuff.

# Example

Throughout the user's guide, we'll use the [Lalonde dataset](http://users.nber.org/~rdehejia/nswdata2.html). It is originally from
Robert Lalonde, "Evaluating the Econometric Evaluations of Training Programs," *American Economic Review*, Vol. 76, pp. 604-620. 1986. This section we'll go through a few ways of loading that data set.

[The example can be found here.](http://nbviewer.ipython.org/url/lendle.github.io/TargetedLearning.jl/user-guide/lalonde_example.ipynb#Loading-the-data-set)

