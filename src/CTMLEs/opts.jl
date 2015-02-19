import MLBase: StratifiedKfold, StratifiedRandomSub
export StratifiedKfold, StratifiedRandomSub

immutable CTMLEOpts
    searchstrategy::SearchStrategy
    QWidx::IntSet
    allgWidx::IntSet
    cvscheme::Type
    cvschemeopts::Tuple
end

"""
The `CTMLEOpts` type stores various options used during the ctmle procedure. When calling `ctmle`, all options have valid defaults.
However, when constructing a `CTMLEOpts` object by calling `CTMLEOpts()` directly, some options must be specified, because the defaults
depend on for example the dimensions of the data set, which are not available to the `CTMLEOpts()` constructor.

**Named arguments**

* `searchstrategy` - Search strategy for choosing next covariates to add to g. See docs for `SearchStrategy`. Default is `ForwardStepwise()`.
* `QWidx` - A collection specifying the column indexes of covariates `w` which should always included in the initial estimate of Q.
The `ctmle` function defaults to all columns of `w`. *If `CTMLEOpts` is called directly, this argument must be specified as the default will error.*
Should be some object which can be converted to an `IntSet` by `IntSet(QWidx)`, for example, an `IntSet`, a range of integers, or a
`Vector{Int}`.
* `allgWidx` - A collection specifying the column indexes of covariates `w` which should always included in all estimates of g. Defaults to intercept only.
Should be some object which can be converted to an `IntSet` by `IntSet(allgWidx)`, for example, an `IntSet`, a range of integers, or a
`Vector{Int}`.
* `cvscheme` - Either `StratifiedRandomSub` or `StratifiedKFold`. This specifies how CV is performed to chose k, the number of covariates to add to g.
Observations are stratified by treatment `a` and outcome `y`.
For details see the documentation on cross validation schemes in MLBase. Note that this should be a type, not an instance of a type.
The `ctmle` function defaults to `StratifiedRandomSub`. *If `CTMLEOpts` is called directly, this argument must be specified as the default will error.*
* `cvschemeopts` - A tuple of options for `cvscheme`. In particular, all arguments except the first to the constructor of `cvscheme`, see documentation in MLBase. If `cvscheme` is
`StratifiedRandomSub`, this should be `(<number observations per subsample>, <number of subsamples>)`. If `cvscheme` is `StratifiedKFold`, this should be `(<number of folds>,)`. 
The `ctmle` function defaults to `(<about 0.9 * n>, 1)`. *If `CTMLEOpts` is called directly, this argument must be specified as the default will error.*
"""
function CTMLEOpts(;searchstrategy::SearchStrategy=ForwardStepwise(),
                   QWidx=:nothing,
                   allgWidx=[1],
                   cvscheme=:nothing,
                   cvschemeopts=:nothing)
    QWidx == :nothing && error("QWidx must be specified when calling CTMLEOpts() directly.")
    cvscheme == :nothing && error("cvscheme must be specified when calling CTMLEOpts() directly.")
    cvschemeopts == :nothing && error("cvshemeopts must be specified when calling CTMLEOpts() directly.")
    if isa(cvschemeopts, Number)
        cvschemeopts = (cvschemeopts, )
    end
    CTMLEOpts(deepcopy(searchstrategy), IntSet(QWidx),  IntSet(allgWidx), cvscheme, cvschemeopts)
end
