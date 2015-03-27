module CTMLEs

using Docile
@document

using Logging
if isdefined(Main, :CTMLE_LOG_LEVEL)
    @eval @Logging.configure(level=$(Main.CTMLE_LOG_LEVEL))
else
    @Logging.configure(level=OFF)
end

export ctmle,

ForwardStepwise,
PreOrdered,
LogisticOrdering,
PartialCorrOrdering,

StratifiedKfold,
StratifiedRandomSub

using StatsBase, MLBase
using ..LReg, ..Common, ..Parameters, ..Qmodels

import ..Qmodels: fluctuate!, defluctuate!, Fluctuation

include("pcor.jl")
include("strategies.jl")

subset{P<:Parameter}(param::P, idx::AbstractVector{Int}) =
    P(map(d -> subset(d, idx), regimens(param))...)

subset(d::StaticRegimen, idx) = d
subset(d::DynamicRegimen, idx) = DynamicRegimen(d.a[idx])

"""
A `Qchunk` holds (a subset of) the data set and corresponding indexes along with a `Qmodel` which can compute predictions for
each observation, a `Parameter` (whos definition may depend on eahc observation) and the value of the emperical risk
for the current `Qmodel`. 
"""
type Qchunk{T<:FloatingPoint}
    q::Qmodel{T}
    W::Matrix{T}
    A::Vector{T}
    Y::Vector{T}
    param::Parameter{T}
    idx::AbstractVector{Int}
    risk::T
end

function make_chunk_subset{T<:FloatingPoint}(logitQnA1::Vector{T}, logitQnA0::Vector{T},
                           W::Matrix{T}, A::Vector{T}, Y::Vector{T},
                           param::Parameter{T}, idx::AbstractVector{Int})
    length(logitQnA1) == length(logitQnA0) == size(W, 1) == length(A) == length(Y) ||
        error(ArgumentError("Input sizes do not match"))
    q = Qmodel(logitQnA1[idx], logitQnA0[idx])
    param = subset(param, idx)
    W = W[idx, :]
    A = A[idx]
    Y = Y[idx]
    q_risk = risk(q, A, Y)
    Qchunk(q, W, A, Y, param, idx, q_risk)
end

StatsBase.nobs(chunk::Qchunk) = nobs(chunk.q)

"""
A `QCV` holds a training `Qchunk` and a validation `Qchunk`, a sequence of estimated gs corresponding to
each fluctuation of the `Qchunk`s, the `SearchStrategy` in use, and indexes of used and unused covariates.
"""
type QCV{T<:FloatingPoint}
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
    gn1_train = predict(g, train.W)
    gn1_val = predict(g, val.W)
    fluc_train = computefluc(train.q, train.param, train_gn1, train.A, train.Y)
    fluc_val = valfluc(fluc_train, val.param, gn1_val, val.A)
    fluctuate!(qcv, g, fluc_train, fluc_val)
end

function makeQCV{T<:FloatingPoint}(logitQnA1::Vector{T}, logitQnA0::Vector{T},
                                   W::Matrix{T}, A::Vector{T}, Y::Vector{T},
                                   param::Parameter{T}, searchstrategy::SearchStrategy,
                                   idx_train::AbstractVector{Int})
    n, p = size(W)
    idx_train = sort(idx_train)
    idx_val = setdiff(1:n, idx_train)
    chunk_train = make_chunk_subset(logitQnA1, logitQnA0, W, A, Y, param, idx_train)
    chunk_val= make_chunk_subset(logitQnA1, logitQnA0, W, A, Y, param, idx_val)
    used_covars = IntSet()
    unused_covars = setdiff(IntSet(1:p), used_covars)
    gseq = LR{T}[]
    QCV(chunk_train, chunk_val, gseq, deepcopy(searchstrategy), used_covars, unused_covars)
end

type CTMLE{T<:FloatingPoint} <: AbstractScalarEstimate
    psi::T
    ic::Vector{T}
    n::Int
    estimand::String
    seasrchstrategy::SearchStrategy
    gseq::Vector{LR{T}}
    function CTMLE(psi, ic, n, estimand, searchstrategy, gseq)
        est = ScalarEstimate(psi, ic)
        new(est.psi, est.ic, n, estimand, searchstrategy, gseq)
    end
end

CTMLE{T<:FloatingPoint}(psi::T, ic::Vector{T}, n::Int, estimand::String,
                        searchstrategy::SearchStrategy, gseq::Vector{LR{T}}) =
    CTMLE{T}(psi, ic, n, estimand, searchstrategy, gseq)

function ctmle{T<:FloatingPoint}(logitQnA1::Vector{T}, logitQnA0::Vector{T},
                                 W::Matrix{T}, A::Vector{T}, Y::Vector{T};
                                 param::Parameter{T}=ATE(),
                                 cvplan = StratifiedRandomSub(zip(A, Y), iround(0.9 * length(Y)), 1),
                                 searchstrategy::SearchStrategy = ForwardStepwise(),
                                 patience::Int=typemax(Int)
                                 )
    n,p = size(W)

    #create vector of QCV objects
    cvqs = [makeQCV(logitQnA1, logitQnA0, W, A, Y, param, searchstrategy, idx_train)
            for idx_train in cvplan]

    #using cross-validation, find best number of steps to take
    steps = find_steps(cvqs, patience=patience)

    #build a chunk with the full data set
    fullchunk = Qchunk(Qmodel(logitQnA1, logitQnA0), W, A, Y, param, 1:n, inf(T))
    gseq = LR{T}[]
    used_covars = IntSet()
    unused_covars = IntSet(1:p)
    #and add covariates steps # of times, saving g fits
    @info("Fitting final estimate of Q")
    for step in 1:steps
        gfit, new_fluc = add_covars!(fullchunk, searchstrategy, used_covars, unused_covars)
        push!(gseq, gfit)
        if length(gseq) == 1
            @info("Step $step of $steps complete, covariate 1 added in new fluctuation.")
        else
            whichfluc = new_fluc? "to current" : "in new"
            whichcovar = setdiff(gseq[end].idx, gseq[end-1].idx)
            @info("Step $step of $steps complete, covariate $whichcovar added $whichfluc fluctuation.")
        end
    end

    q=fullchunk.q
    CTMLE(applyparam(param, q, A, Y)..., nobs(q), estimand(param), searchstrategy, gseq)

end


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

Returns most recent `gfit` object and a boolean which is `true` if the most recent covariates were added to a new fluctuation
or `false` if the previous fluctuation was updated with new covariates.

"""
function find_steps{T<:FloatingPoint}(qcvs::Vector{QCV{T}}; patience::Int=typemax(Int))
    done = false
    steps = 0
    best_steps = 0
    best_risk = Inf

    @info("Choosing number of steps via cross validation")
    while !done
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
            done = true
        end
        if all(map(qcv -> isempty(qcv.unused_covars), qcvs))
            @info("Crossvalidation completed after $steps steps.")
            done = true
        end
    end
    best_steps
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
        @debug("Initial fluctuation")
        push!(used_covars, 1)
        delete!(unused_covars, 1)
        gfit = lreg(chunk.W, chunk.A, subset=1:1)
        fluc = computefluc(chunk.q, chunk.param, predict(gfit, chunk.W), chunk.A, chunk.Y)
        fluctuate!(chunk.q, fluc)
        chunk.risk = risk(chunk.q,chunk.A, chunk.Y)
        return gfit, true
    else
        @debug("Adding covariate(s)")
        q_risk, gfit, new_fluc = add_covars!(searchstrategy, chunk.q, chunk.param, chunk.W, chunk.A, chunk.Y,
                                   used_covars, unused_covars, chunk.risk)
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
    gn1_val = predict(gfit, val.W)
    if !new_fluc
        defluctuate!(val.q)
    end
    fluctuate!(val.q, valfluc(lastfluc(train.q), val.param, gn1_val, val.A))
    val.risk = risk(val.q, val.A, val.Y)

    qcv
end

end # module


