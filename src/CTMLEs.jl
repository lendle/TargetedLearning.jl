@reexport module CTMLEs

using NumericExtensions, StatsBase, MLBase, Devectorize

import StatsBase.predict, StatsBase.predict!, NumericExtensions.evaluate

export CTMLE, ctmle

const debug = 0

include("LReg.jl")
using .LReg

include("pcor.jl")
include("atefunctors.jl")
include("Qmodel.jl")
include("strategies.jl")
include("opts.jl")

type CTMLE
    psi::Float64
    q::Qmodel
    opts::CTMLEOpts
end

function Base.show(io::IO, obj::CTMLE)
    println(io, "Estimate: $(obj.psi)")
    print(io, "Search strategy: ")
    print(io, "\n")
    show(io, obj.opts.searchstrategy)
    show(io, obj.q)
end


function build_Q!(qfit::Qmodel, dat, valdat=:none; k=typemax(Int), opts=CTMLEOpts())
    #qfit should just be the inital fit of Qbar, not fluctuated.
    #dat is the tuple (w, a, y) of the training data.
    w, a, y = dat

    #valdat is :none or a tuple (w_val, a_val, y_val) of validation data.

    #set up storage for risk estimates
    train_risk = Float64[]
    val_risk = Float64[]

    #initialize the search strategy object
    init!(opts.searchstrategy)

    #fit initial g with intercept only
    ginit = sslreg(w, a, IntSet(1))

    #fluctuate qfit with intercept-only g
    fluctuate!(qfit, ginit, dat...)

    used_covars = IntSet(1) #indicies for covariates used in g, starts as intercept only
    unused_covars = IntSet(2:size(w, 2)) #indecies for covariates not yet in g

    while ! isempty(unused_covars) && length(used_covars) < k
        push!(train_risk, risk(qfit, w, a, y))
        if valdat != :none
            push!(val_risk, risk(qfit, dat..., pen=false))
        end
        add_covar!(opts.searchstrategy, qfit, w, a, y, used_covars, unused_covars, train_risk[end])
    end

    (train_risk, val_risk)
end

function ctmle(w, a, y, QWidx = 1:size(w, 2), opts::CTMLEOpts=CTMLEOpts())
    #QWidx specifies the indexes of the covars in W to be used to fit Q

    n = length(y)
    @assert n == length(a) == size(w, 1)

    #it is assumed that the first column of w is 1.0 for an intercept
    @assert all(w[:, 1] .== 1.0)

    QWAidx = union(QWidx, size(w,2)+ 1)

    train_idx = collect(StratifiedRandomSub(zip(a, y), iround(0.9 * n), 1))[1]
    val_idx = setdiff(1:n, train_idx)

    train_dat = (w[train_idx, :], a[train_idx], y[train_idx])
    val_dat = (w[val_idx, :], a[val_idx], y[val_idx])

    #fit initial Q and build fluctuations on training data
    train_qfit = Qmodel(sparselreg([train_dat[1] train_dat[2]], train_dat[3], QWAidx))
    debug > 0 && info("building training Q")
    train_risk, val_risk = build_Q!(train_qfit, train_dat, val_dat, opts=opts)
    best_k_risk, best_k = findmin(val_risk)

    #fit initial Q and build fluctuations on validation data, adding best_k terms
    qfit = Qmodel(sparselreg([w a], y, QWAidx))
    debug > 0 && info("building full Q")
    build_Q!(qfit, (w, a, y), k = best_k, opts=opts)

    QnA1 = predict(qfit, w, ones(n), :prob)
    QnA0 = predict(qfit, w, zeros(n), :prob)

    CTMLE(mean(QnA1) - mean(QnA0), qfit, opts)
end

ctmle(w, a, y, QWidx = 1:size(w, 2); opts...) = ctmle(w, a, y, QWidx, CTMLEOpts(;opts...))

end # module
