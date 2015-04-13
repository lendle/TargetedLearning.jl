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

To download the example data set, run
```julia
include(joinpath(Pkg.dir("TargetedLearning"), "examples", "fetchdata.jl"))
```
This will create a CSV in ```Pkg.dir("TargetedLearning")`/examples/data/lalonde.csv``.
This only needs to be run once.

## Using readcsv

The simplest way to read the data set is with `readcsv`. The CSV file we have has a header, so we'll pass `header=true`, and we'll get back a tuple containing a matrix with the numeric data in it (`readcsv` can figure out that it's all `Float64`s automatically) and a matrix with one row containing column names.

```jlcon
julia> dsfname = joinpath(Pkg.dir("TargetedLearning"), "examples", "data", "lalonde.csv")
"/home/sam/.julia/v0.3/TargetedLearning/examples/data/lalonde.csv"

julia> # the ; at the end of the next line keeps the output from printing in the REPL
       dat, colnames = readcsv(dsfname, header=true);

julia> #lets inspect colnames
       colnames
1x9 Array{String,2}:
 "treatment"  "age"  "education"  "black"  …  "married"  "nodegree"  "RE75"  "RE78"

julia> #lets convert it to a vector instead of a matrix with one row
       colnames = reshape(colnames, size(colnames, 2))
9-element Array{String,1}:
 "treatment"
 "age"
 "education"
 "black"
 "hispanic"
 "married"
 "nodegree"
 "RE75"
 "RE78"
```

`treatment` is obviously the treatment variable. The outcome variable is `RE78` (earnings in 1978), and `RE75` is the pretreatment value at baseline. The others are potential baseline confounders. Check the link above for more information about the dataset.  We want to slice the matrix `dat` up to extract the treatment and outcome variable. Julia uses square brackets for indexing. The first dimension of a matrix is rows and the second is columns like R. If you want everything in one dimension, you put `:` (where as in R you can leave that dimension empty). You can index with booleans, integers, vectors of integers, ranges and some other things. Here are some examples:

```
julia> #we know column 1 is treatment, so get it directly, and ask for all rows with :
       treatment = dat[:, 1];

julia> #suppose instead we want to find the position in colnames that has the value
       #"treatment", but we don't know it's the first one. There are a couple of ways
       #to do that.
       #(.== and operators starting with `.` in general indicate that we want to do
       #   an element wise operation)
       treatment_take2 = dat[:, colnames .== "treatment", 1];

julia> #the last column is the outcome so we can use the keyword `end`
       outcome = dat[:, end];

julia> #we can also use `end` in arithmetic, e.g.
       outcome_prebaseline = dat[:, end-1];

julia> #baseline covariates are everything between the first and last column, so we could use a range:
       baseline_covars = dat[:, 2:end-1];

```

<!---
_Skip this if you want to!_ As a bonus example, suppose we want to search for some pattern in the column names, for example, those starting with "RE". We can use a simple regular expression to find them.
```
julia> re_idxs = find(name -> ismatch(r"^RE", name), colnames)
2-element Array{Int64,1}:
 8
 9

julia> re_vars = dat[:, re_idxs];
```
There's kind of a lot going on here. `find` can do a few things (see `?find`). In this case, it's taking a function that returns true or false as it's first argument, and then calls that function on each element of the second argument (`colnames`). `find` then returns the indexes of elements for which the function evaluated to true. The `->` syntax constructs an anonymous function. `name -> ismatch(r"^RE", name)` is a function that takes a single parameter called `name`, and returns the result of `ismatch(r"^RE", name)`. `ismatch` is a function that takes a regular expression and a string, and returns true if that regex matches the string.  `r"pattern"` is how you make a regex in Julia. The pattern `^RE` is a pattern which will match on any string that starts with "RE".  We see that we wind up with `re_idxs` having the indexes `8` and `9` corresponding to the last two columns.
-->

## Using DataFrame

The `readtable` function returns a `DataFrame` which allows for indexing by column names.

```jlcon
julia> using DataFrames #like library in R

julia> dsfname = joinpath(Pkg.dir("TargetedLearning"), "examples", "data", "lalonde.csv")
"/home/sam/.julia/v0.3/TargetedLearning/examples/data/lalonde.csv"

julia> df = readtable(dsfname);

julia> #check the column names:
       names(df)
9-element Array{Symbol,1}:
 :treatment
 :age
 :education
 :black
 :hispanic
 :married
 :nodegree
 :RE75
 :RE78
```

Data frames in Julia are indexed by symbols instead of strings. Symbols are no entirely unlike strings, and are created in julia with `:symbolname` or `Symbol("symbolname")`.

Now let's get the treatment and outcome variables out of `df`.

```jlcon
julia> treatment = df[:treatment];

julia> outcome = df[:RE78];

julia> #look at the first few values of outcome
       head(outcome)
6-element DataArray{Float64,1}:
  9930.05
  3595.89
 24909.5
  7506.15
   289.79
  4056.49
```
We see that variables in a data frame are actually `DataArray`s instead of regular Julia `Array`s. The functions in TargetedLearning.jl currently only work with `Arrays` of floating point numbers, so we'll convert them.

```jlcon
julia> treatment = convert(Array{Float64}, treatment);

julia> outcome = convert(Array{Float64}, outcome);
```

We can also index into the data frame using ranges, just like regular matrixes (but we only index on columns). Let's get the baseline covariates. When you get more than one column out of a data frame, you get back another data frame. For some reason that I do not know, `convert` won't work for us here, but `array` will get us what we need (a Julia array instead of a DataFrame).

```jlcon
julia> baseline_covars = array(df[2:end-1]);
```


### Formulas

One nice thing about DataFrames.jl is that is has support for formulas. They are similar to R's formulas, but there are some differences. Some packages, like GLM.jl take data frames with formulas as input. Those packages can be used for computing initial estimates, but TargetedLearning.jl does not support formulas and DataFrames directly.  However, you can use DataFrames.jl's functionality to take a DataFrame and a formula and get a numeric design matrix based on a formula.

For example, suppose we'd like to include an age squared term and an interaction term between education and marital status. It looks like polynomial terms aren't implemented currently, so you'll have to manually make those terms.
```
julia> #create age squared
       df[:age2] = df[:age] .* df[:age];

julia> #suppress the intercept with -1
       fm = treatment ~ -1 + age + age2 + education + black + hispanic + married + nodegree + RE75 + married&nodegree
Formula: treatment ~ -1 + age + age2 + education + black + hispanic + married + nodegree + RE75 + married & nodegree

julia> #take the field named `m` from the created ModelMatrix object
       expanded_baseline_covars = ModelMatrix(ModelFrame(fm, df)).m;

```

It's clunky, but will get the job done. More detailed documentation is [here](http://dataframesjl.readthedocs.org/en/latest/formulas.html).

### Missing data and categorical data

[DataFrames.jl](http://dataframesjl.readthedocs.org/en/latest/) and [DataArrays.jl](https://github.com/JuliaStats/DataArrays.jl) have ways of handling both missing data and categorical data. TargetedLearning.jl does not, so you'll have to deal with those issues ahead of time. The documentation for those packages has more information on both.
