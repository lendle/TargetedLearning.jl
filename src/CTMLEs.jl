module CTMLEs

using Logging
if isdefined(Main, :CTMLE_LOG_LEVEL)
    @eval @Logging.configure(level=$(Main.CTMLE_LOG_LEVEL))
else
    @Logging.configure(level=WARNING)
end

export ctmle,

SearchStrategy,
ForwardStepwise,
PreOrdered,
OrderingStrategy,
LogisticOrdering,
PartialCorrOrdering,
HDPSOrdering,

StratifiedKfold,
StratifiedRandomSub,

flucinfo

using StatsBase, MLBase
using ..LReg, ..Common, ..Parameters, ..Qmodels

import ..Qmodels: fluctuate!, defluctuate!, Fluctuation

include("pcor.jl")
include("strategies.jl")

typealias ScalarOrVec{T} Union{T, Vector{T}}

subset{P<:Parameter}(param::P, idx::AbstractVector{Int}) =
    P(map(d -> subset(d, idx), regimens(param))...)

subset(d::StaticRegimen, idx) = d
subset(d::DynamicRegimen, idx) = DynamicRegimen(d.a[idx])

"""
A `Qchunk` holds (a subset of) the data set and corresponding indexes along with a `Qmodel` which can compute predictions for
each observation, a `Parameter` (whos definition may depend on each observation) and the value of the emperical risk
for the current `Qmodel`.
"""
type Qchunk{T<:AbstractFloat}
    q::Qmodel{T}
    W::Matrix{T}
    A::Vector{T}
    Y::Vector{T}
    param::Parameter{T}
    idx::AbstractVector{Int}
    gbounds::Vector{T}
    penalize::Bool
    risk::T
end

function make_chunk_subset{T<:AbstractFloat}(logitQnA1::Vector{T}, logitQnA0::Vector{T},
                           W::Matrix{T}, A::Vector{T}, Y::Vector{T},
                           param::Parameter{T}, idx::AbstractVector{Int}, gbounds::Vector{T},
                           penalize::Bool)
    length(logitQnA1) == length(logitQnA0) == size(W, 1) == length(A) == length(Y) ||
        error(ArgumentError("Input sizes do not match"))
    q = Qmodel(logitQnA1[idx], logitQnA0[idx])
    param = subset(param, idx)
    W = W[idx, :]
    A = A[idx]
    Y = Y[idx]
    #Can't compute penalized risk if q isn't fluctuated yet,
    #so set to Inf
    q_risk = penalize? convert(T, Inf) : risk(q, A, Y, param, false)
    Qchunk(q, W, A, Y, param, idx, gbounds, penalize, q_risk)
end

StatsBase.nobs(chunk::Qchunk) = nobs(chunk.q)

"""
A `QCV` holds a training `Qchunk` and a validation `Qchunk`, a sequence of estimated gs corresponding to
each fluctuation of the `Qchunk`s, the `SearchStrategy` in use, and indexes of used and unused covariates.
"""
type QCV{T<:AbstractFloat}
    chunk_train::Qchunk{T}
    chunk_val::Qchunk{T}
    gseq::Vector{LR{T}}
    searchstrategy::SearchStrategy
    used_covars::IntSet
    unused_covars::IntSet
end

"""
Removes the last fluctuation from a `QCV`

** Arguments **

* `qcv` - A `QCV`

Returns last estimated g and last fluctuation from the training and validation chunks.
"""
function defluctuate!(qcv::QCV)
    (pop!(qcv.gseq),
     defluctuate!(qcv.chunk_train.q),
     defluctuate!(qcv.chunk_val.q))
end

"""
Adds a previously calculated fluctuation to a `QCV`

** Arguments **

* `qcv` - A `QCV`
* `g` - A `LR`, estimate of g
* `fluc_train` - fluctuation for the training set
* `fluc_val` - fluctuation for the validation set

Returns training and validation fluctuations.
"""
function fluctuate!(qcv::QCV, g::LR, fluc_train::Fluctuation, fluc_val::Fluctuation)
    push!(qcv.gseq, g)
    fluctuate!(qcv.chunk_train, fluc_train)
    fluctuate!(qcv.chunk_val, fluc_val)
    (fluc_train, fluc_val)
end

"""
Adds a fluctuation to a `QCV` given an estimated g

** Arguments **

* `qcv` - A `QCV`
* `g` - A `LR`, estimate of g

** Details **

A new fluctuation for the training chunk is calculated along with a corresponding flucutation for the validation
chunk, based on the estiamted `g`, then the `QCV` is updated.

Returns training and validation fluctuations.
"""
function fluctuate!(qcv, g::LR)
    train = qcv.train_chunk
    val = qcv.val_chunk
    gn1_train = bound!(predict(g, train.W), train.gbounds)
    gn1_val = bound!(predict(g, val.W), val.gbounds)
    fluc_train = computefluc(train.q, train.param, gn1_train, train.A, train.Y)
    fluc_val = valfluc(fluc_train, val.param, gn1_val, val.A)
    fluctuate!(qcv, g, fluc_train, fluc_val)
end

function makeQCV{T<:AbstractFloat}(logitQnA1::Vector{T}, logitQnA0::Vector{T},
                                   W::Matrix{T}, A::Vector{T}, Y::Vector{T},
                                   param::Parameter{T}, searchstrategy::SearchStrategy,
                                   idx_train::AbstractVector{Int}, gbounds::Vector{T},
                                   penalize::Bool)
    n, p = size(W)
    idx_train = sort(idx_train)
    idx_val = setdiff(1:n, idx_train)
    chunk_train = make_chunk_subset(logitQnA1, logitQnA0, W, A, Y, param, idx_train, gbounds, penalize)
    chunk_val= make_chunk_subset(logitQnA1, logitQnA0, W, A, Y, param, idx_val, gbounds, false)
    used_covars = IntSet()
    unused_covars = setdiff(IntSet(1:p), used_covars)
    gseq = LR{T}[]
    QCV(chunk_train, chunk_val, gseq, deepcopy(searchstrategy), used_covars, unused_covars)
end

immutable FluctuationInfo
    steps::Int
    covar_order::Vector{Vector{Int}}
    new_flucseq::Vector{Bool}
    added_each_fluc::Vector{Vector{Int}}
    function FluctuationInfo(steps::Int, covar_order::Vector{Vector{Int}},
                             new_flucseq::Vector{Bool},
                             added_each_fluc::Vector{Vector{Int}})
        steps == length(covar_order) == length(new_flucseq) || throw(ArgumentError("steps should equal length of covar_order and new_flucseq"))
        length(added_each_fluc) == sum(new_flucseq) || throw(ArgumentError("length(added_each_fluc) not equal to number of new flucs"))
        new(steps, covar_order, new_flucseq, added_each_fluc)
    end
end

function FluctuationInfo{T}(steps::Int, gseq::Vector{LR{T}}, new_flucseq::Vector{Bool})
    new_flucseq[1] || throw(ArgumentError("new_flucseq[1] should be true"))
    first_covars = Vector{Int}[convert(Vector{Int}, gseq[1].idx)]::Vector{Vector{Int}}
    added_covar_order = [setdiff(a.idx, b.idx)::Vector{Int} for (a, b) in zip(gseq[2:end], gseq[1:end-1])]::Vector{Vector{Int}}
    covar_order = [first_covars; added_covar_order]::Vector{Vector{Int}}
    start_new_fluc = find(new_flucseq)
    end_fluc = [find(new_flucseq)[2:end] - 1; length(new_flucseq)]
    added_each_fluc = [vcat(covar_order[st:en]...)::Vector{Int}
                       for (st, en) in zip(start_new_fluc, end_fluc)]
    FluctuationInfo(steps, covar_order, new_flucseq, added_each_fluc)
end

function Base.show(io::IO, flucinfo::FluctuationInfo)
    println(io, "Fluctuation info:")
    println(io, "Covariates added in $(flucinfo.steps) steps to $(length(flucinfo.added_each_fluc)) fluctuations.")
    for (i, new_covars) in enumerate(flucinfo.added_each_fluc)
        println(io, "Fluc $i covars added: $new_covars")
    end
end

type CTMLE{T<:AbstractFloat} <: AbstractScalarEstimate
    psi::T
    ic::Vector{T}
    n::Int
    estimand::AbstractString
    seasrchstrategy::SearchStrategy
    flucinfo::FluctuationInfo
    function CTMLE(psi, ic, n, estimand, searchstrategy, flucinfo)
        est = ScalarEstimate(psi, ic)
        new(est.psi, est.ic, n, estimand, searchstrategy, flucinfo)
    end
end

CTMLE{T<:AbstractFloat}(psi::T, ic::Vector{T}, n::Int, estimand::AbstractString,
                        searchstrategy::SearchStrategy, flucinfo::FluctuationInfo) =
    CTMLE{T}(psi, ic, n, estimand, searchstrategy, flucinfo)


"""
Performs colaborative targeted minimum loss-based estimation

** Arguments **

* `logitQnA1` - Vector of length n of initial estimates logit(\bar{Q}_n(1, W_i))
* `logitQnA0` - Vector of length n of initial estimates logit(\bar{Q}_n(0, W_i))
* `W` - Matrix of covariates to be potentially used to estimate g. n rows.
        The first column should be all ones for an intercept.
* `A` - Vector of length n of treatments, 0.0 or 1.0
* `Y` - Vector of length n of outcomes, bounded between 0 and 1.

** Keyword Arguments **

* `param` - Target parameter. See ?ATE or ?Mean. Default is ATE()
* `gbounds` - values for bounding g. Estimates of g will be bounded by `extrema(gbounds)`.
   To only bound away from 0 (or 1), set to e.g. `[0.01, 1.0]` (or `[0.0, 0.99]`).
   To turn off bounding all together, set to `[0.0, 1.0]`. Defaults to `[0.01, 0.99]`.
* `penalize_risk` - should a penalized risk be used when choosing covariates to estimate g?
   Defaults to false.
* `searchstrategy` - Strategy or a vector of strategies for adding covariates to estimates of g.
   See ?SearchStrategy. If a vector, then the best strategy is selected via cross-validation.
   Default is ForwardStepwise()
* `cvplan` - An iterator of vectors of Int indexes of observations in each training set for
   cross validaiton. StratifiedKfold and StratifiedRandomSub from the MLBase package are useful
   here. Defaults to 10 fold CV stratifying by treatment for outcomes with more than 2 levels,
   or by treatment and outcome for binary outcomes.
* `patience` - For how many steps should CV continue after finding a local optimum? Defaults to typemax(Int)

"""
function ctmle{T<:AbstractFloat,
               S<:SearchStrategy}(logitQnA1::Vector{T}, logitQnA0::Vector{T},
                                  W::Matrix{T}, A::Vector{T}, Y::Vector{T};
                                  param::Parameter{T}=ATE(),
                                  gbounds::Vector{T}=[0.01, 0.99],
                                  penalize_risk::Bool=false,
                                  searchstrategy::ScalarOrVec{S} = ForwardStepwise(),
                                  cvplan = StratifiedKfold(length(unique(Y))<3? zip(A, Y) : A, 10),
                                  patience::Int=typemax(Int)
                                  )
    n,p = size(W)
    n == length(logitQnA1) ==
        length(logitQnA0) ==
        length(A) ==
        length(Y) ||
        throw(ArgumentError("logitQnA1, logitQnA0, A, and Y should have length equal to the number of rows of W."))

    all(W[:, 1] .== 1) || throw(ArgumentError("The first column of W should be all ones."))

    strategies = isa(searchstrategy, Vector)? searchstrategy : [searchstrategy]

    #create vector of QCV objects
    cvqs = [makeQCV(logitQnA1, logitQnA0, W, A, Y, param, searchstrategy,
                    idx_train, gbounds, penalize_risk)::QCV{T}
            for idx_train in cvplan, searchstrategy in strategies]

    #using cross-validation, find best number of steps to take
    best_strat_idx, best_steps = find_strat_steps(cvqs, patience=patience)

    best_strategy = strategies[best_strat_idx]

    #build a chunk with the full data set
    fullchunk = Qchunk(Qmodel(logitQnA1, logitQnA0), W, A, Y, param, 1:n, gbounds, penalize_risk, convert(T, Inf))
    gseq = LR{T}[]
    new_flucseq = Bool[]
    used_covars = IntSet()
    unused_covars = IntSet(1:p)
    #and add covariates steps # of times, saving g fits
    @info("Fitting final estimate of Q")
    for step in 1:best_steps
#         @debug("used_covars: $used_covars")
#         @debug("unused_covars: $unused_covars")
        gfit, new_fluc = add_covars!(fullchunk, best_strategy, used_covars, unused_covars)
        push!(gseq, gfit)
        push!(new_flucseq, new_fluc)
        @info(if length(gseq) == 1
                  "Step $step of $best_steps complete, covariate [1] added in new fluctuation."
              else
                  whichfluc = new_fluc? "in new" : "to current"
                  whichcovars = setdiff(gseq[end].idx, gseq[end-1].idx)
                  "Step $step of $best_steps complete, covariates $whichcovars added $whichfluc fluctuation."
              end)
    end

    #create flucinfo
    flucinfo = FluctuationInfo(best_steps, gseq, new_flucseq)

    q=fullchunk.q
    CTMLE(applyparam(param, q, A, Y)..., nobs(q), estimand(param), best_strategy, flucinfo)

end

flucinfo(est::CTMLE) = est.flucinfo

"""
Determines the number of times covariates should be added using cross validation.

** Arguments **

* `qcvs` - A vector of one or more `QCV`s

** Keyword Arguments **

* `patience` - An integer. How many more steps should we try after finding a local minimum?
Defaults to a big number.

** Details **

Adds covariates to each `qcv` and computes the mean risk on validation chunks.
If minimum validation risk occured more than `patience` steps ago, we stop looking.
So for example if `patience` is `2`, we try for two more steps after finding a local minimum to see
if we can do better.
If `patience` is `0`, we stop as soon as risk starts increasing.
If `pateince` is negative, we stop after 1 step, which is probably not a good idea.

Returns best risk and number of steps.

"""
function find_steps{T<:AbstractFloat}(qcvs::Vector{QCV{T}}; patience::Int=typemax(Int))
    notdone = true
    steps = 0
    best_steps = 0
    best_risk = Inf

    @info("Choosing number of steps via cross validation")
    while notdone
        steps += 1
        map(add_covars!, qcvs)
        val_risk = mean(map(qcv -> qcv.chunk_val.risk, qcvs))
        if val_risk < best_risk
            best_risk = val_risk
            best_steps = steps
        end
        @info("Step $steps validation risk: $val_risk. Best risk so far: $best_risk at step $best_steps")
        if steps - best_steps > patience
            @info("No improvement in $patience steps, stopping at step $steps.")
            notdone = false
        end
        if all(map(qcv -> isempty(qcv.unused_covars), qcvs))
            @info("Crossvalidation completed after $steps steps.")
            notdone = false
        end
    end
    (best_risk, best_steps)
end

function find_strat_steps{T<:AbstractFloat}(qcvs::Matrix{QCV{T}}; patience::Int=typemax(Int))
    risks, steps = mapslices(qcvs,1) do qcvcol
        find_steps(qcvcol, patience=patience)
    end |> x -> zip(x...)

    best_strat_idx = findmin(risks)[2]

    best_steps = steps[best_strat_idx]
    (best_strat_idx, best_steps)
end

"""
Adds next covariate(s) to a Qchunk based on a search strategy

** Arguments **

* `chunk` - a `Qchunk`
* `searchstrategy` - a `SearchStrategy`
* `used_covars` - indexes of covariates already included in g
* `unused_covars` - indexes of covariates not yet included in g

** Details **

If `used_covars` is emtpy (because this is the first time `add_covars!` has been called on this chunk,
a fluctuation based on an intercept only model is added to Q.
Otherwise, next covariates are added where the order is determined by `searchstrategy`

Returns most recent `gfit` object and a boolean which is `true` if the most recent covariates were added to a new fluctuation
or `false` if the previous fluctuation was updated with new covariates.

"""
function add_covars!(chunk::Qchunk, searchstrategy::SearchStrategy, used_covars, unused_covars)
    if isempty(used_covars)
#         @debug("Initial fluctuation")
        push!(used_covars, 1)
        delete!(unused_covars, 1)
        gfit = lreg(chunk.W, chunk.A, subset=1:1)
        fluc = computefluc(chunk.q, chunk.param, bound!(predict(gfit, chunk.W), chunk.gbounds), chunk.A, chunk.Y)
        fluctuate!(chunk.q, fluc)
        chunk.risk = risk(chunk.q,chunk.A, chunk.Y, chunk.param, chunk.penalize)
        return gfit, true
    else
#         @debug("Adding covariate(s)")
        abc = add_covars!(searchstrategy, chunk.q, chunk.param, chunk.W, chunk.A, chunk.Y,
                          used_covars, unused_covars, chunk.risk, chunk.gbounds, chunk.penalize)
        q_risk, gfit, new_fluc = abc
        chunk.risk=q_risk
        return gfit, new_fluc
    end
end

"""
Adds next covariate(s) to a QCV.

** Arguments **

* `qcv` - a `QCV`

** Details **
If `qcv` has no unused covariates, none are added.  New covariates are first added to the training chunk,
`qcv.train`. Then the validation chunk fluctuated using the newest `gfit`.
Returns updated `qcv`.

"""
function add_covars!(qcv::QCV)
    if isempty(qcv.unused_covars)
        @debug("No covariates to add")
        return qcv
    end
    train = qcv.chunk_train
    val = qcv.chunk_val
    #find next fluctuation for q_train
    # update q_val, which dpeends on g fit on q_train, and fluc

    #add next covar to training chunk and store gfit
    gfit, new_fluc = add_covars!(train, qcv.searchstrategy, qcv.used_covars, qcv.unused_covars)
    push!(qcv.gseq, gfit)

    #add newest fluctuation to val chunk and compute risk
    val = qcv.chunk_val
    gn1_val = bound!(predict(gfit, val.W), val.gbounds)
    if !new_fluc
        defluctuate!(val.q)
    end
    fluctuate!(val.q, valfluc(lastfluc(train.q), val.param, gn1_val, val.A))
    val.risk = risk(val.q, val.A, val.Y, val.param, false)

    qcv
end

function bound!{T<:AbstractFloat}(gn1::Vector{T}, gbounds::Vector{T})
    mn, mx = extrema(gbounds)
    mn < 0.5 || ArgumentError("minimum(gbounds) should be < 0.5")
    mx > 0.5 || ArgumentError("maximum(gbounds) should be > 0.5")

    mn > 0.0 || mx < 1.0 || return gn1

    for i in 1:length(gn1)
        @inbounds gn1[i] = max(mn, min(mx,gn1[i]))
    end
    gn1
end

function risk{T<:AbstractFloat}(q::Qmodel{T}, A::Vector{T}, Y::Vector{T}, param::Parameter{T}, penalize::Bool)
    total_loss = 0.0
    length(Y) == nobs(q) || error()
    lp = linpred(q, A)
    for i in 1:nobs(q)
      total_loss += LReg.loss(Y[i], lp[i])
    end
    q_risk = total_loss/nobs(q)


    if penalize
        ic = applyparam(param, q, A, Y)[2]
        pen = var(ic) / length(A)
    else
        pen = 0.0
    end
    q_risk + pen
end

end # module
