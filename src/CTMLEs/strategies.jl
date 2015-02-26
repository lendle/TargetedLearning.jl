module Strategies

using Docile
@document

export SearchStrategy, ForwardStepwise, PreOrdered
export OrderingStrategy, LogisticOrdering, PartialCorrOrdering, HDPSOrdering

abstract OrderingStrategy
type LogisticOrdering <: OrderingStrategy end
type PartialCorrOrdering <: OrderingStrategy end
type HDPSOrdering <: OrderingStrategy end

"""
Defines how the next covariate added to g is chosen

* `ForwardStepwise` - default, tries each unused covariate to find the one that helps the most. Slow, O(p^2) time for p covariates.
* `PreOrdered` - Orders covariates once then adds them in that order. Takes O(p) time for p covariates. Specify with `PreOrdered(ordering)` where `ordering` is
    * `LogisticOrdering()` - ranks covariates based on reduction in log likelihood with intital Q as an offset if added one at a time. Basically the first step of `ForwardStepwise`
    * `PartialCorrOrdering()` - ranks covariates based on partial correlation of each covariate and the residual (y - predicted y from initial Qbar) given a
    * `HDPSOrdering()` - ranks covariates based on bias reduction potential as in the hdps procedure. This ignores intial Qbar.
"""
abstract SearchStrategy
type ForwardStepwise <: SearchStrategy end
type PreOrdered <: SearchStrategy
    ordering::OrderingStrategy
    katatime::Int
    covar_order::Vector{Int}
    PreOrdered(ordering::OrderingStrategy, katatime=1) = new(ordering, katatime, Array(Int, 0))
end
Base.show(io::IO, strat::PreOrdered) = print(io, "Preordered($(strat.ordering))")
end

using .Strategies

#strategies may store some state and we need to clear it out sometimes
init!(strat::SearchStrategy) = strat
function init!(strat::PreOrdered)
    strat.covar_order = Int[]
    strat
end


## add_covar! adds a covariate to the fluctuations of qfit based on a SearchStrategy, the first argument
## The next added covar may be added to the current clever covariate by removing the last fluctation from Q
## ("defluctuating") and replacing it with a new fluctuation, where g is fit on the next covariate.
## If that makes loss for Q worse, then the original last fluctation is replaced (by defluctuating then "refluctuating")
## and an additional fluctuation is added with g based the previous covariates and one additional.

function add_covar!(::ForwardStepwise, qfit, w, a, y, used_covars, unused_covars, prev_risk)
    #set up an array with length = max index of unused_covars initialized to Inf
    next_covar_risk = fill(Inf, maximum(unused_covars))

    g_and_flucs = Dict{IntSet, (LR, LR)}()

    debug[1] > 0 && info("unused_covars: $unused_covars")
    debug[1] > 0 && info("used_covars: $used_covars")

    #Remove most recent fluctuation (and save for later use)
    #So that we can add one covar to the current fluctuation to see if it works
    g_now, fluc_now = defluctuate!(qfit)

    #compute g fluctuations and risks
    gfits_flucs_risks = pmap(unused_covars) do j
        current_covars = union(used_covars, IntSet(j))
        #estimate g with additional covariate j
        gfit = lreg(w, a, subset=collect(current_covars))
        #compute fluctuation
        fluc = computefluc(qfit, gfit, w, a, y)
        #compute risk on a copy of q
        r = risk(fluctuate(qfit, gfit, fluc), w, a, y)
        (gfit, fluc, r)
    end

    #store fluctuations and risks computed in parallel in a nice way
    for (idx, j) in enumerate(unused_covars)
        current_covars = union(used_covars, IntSet(j))
        gfit, fluc, r = gfits_flucs_risks[idx]
        next_covar_risk[j] = r
        g_and_flucs[current_covars] = (gfit, fluc)
    end
    best_risk, best_j = findmin(next_covar_risk)

    if best_risk < prev_risk
        #adding one covariate to current fluc does better, then use that fluctuation
        push!(used_covars, best_j)
        delete!(unused_covars, best_j)
        fluctuate!(qfit, g_and_flucs[used_covars]...)
    else
        #otherwise, keep current fluctuation find best new fluctuation adding a
        #single covariate

        fluctuate!(qfit, g_now, fluc_now) #reuse previous best fluctuation

        next_covar_risk = fill(Inf, maximum(unused_covars))
        for j in unused_covars
            current_covars = union(used_covars, IntSet(j))
            #gfit = sslreg(w, a, current_covars)
            gfit = g_and_flucs[current_covars][1]
            fluctuate!(qfit, gfit, w, a, y)
            next_covar_risk[j] = risk(qfit, w, a, y)
            if isnan(next_covar_risk[j])
                f = open("blah", "w")
                serialize(f, qfit)
                close(f)
            end
            g_and_flucs[current_covars] = defluctuate!(qfit)
        end
        debug[1] > 0 && info("next_covar_risk: $next_covar_risk")
        best_risk, best_j = findmin(next_covar_risk)

        # if best_risk > best_risk
        #     info("!!!Best score for new covar is worse than score without. Continuing anyway")
        # end
        # verbose >= 2 && info("Best updated covar score: $best_risk")
        # verbose >= 1 && info("Adding covar $best_j to new clever covar")
        debug[1] > 0 && info("best_j: $best_j, unused_covars: $unused_covars")
        push!(used_covars, best_j)
        delete!(unused_covars, best_j)
        #fluctuate!(qfit, sslreg(w, a, used_covars), w, a, y)
        try fluctuate!(qfit, g_and_flucs[used_covars]...)
        catch e
            info("best_j: $best_j, unused_covars: $unused_covars")
            rethrow(e)
        end
    end
end

function add_covar!(strategy::PreOrdered, qfit, w, a, y, used_covars, unused_covars, prev_risk)
    if isempty(strategy.covar_order)
        append!(strategy.covar_order, order_covars(strategy.ordering, qfit, w, a, y, unused_covars))
    end


    ordered_unused_covars = filter(x -> x ∉ used_covars, strategy.covar_order)
    next_covars = length(ordered_unused_covars) >= strategy.katatime ?
        ordered_unused_covars[1:strategy.katatime] :
        ordered_unused_covars


    g_old, fluc_old = defluctuate!(qfit)

    union!(used_covars, next_covars)
    setdiff!(unused_covars, next_covars)

    g_fit = lreg(w, a, subset=collect(used_covars))

    fluctuate!(qfit, g_fit, w, a, y)

    if risk(qfit, w, a, y) < prev_risk
        return
    else
        defluctuate!(qfit)
        fluctuate!(qfit, g_old, fluc_old)
        fluctuate!(qfit, g_fit, w, a, y)
    end
end

function order_covars(ordering::LogisticOrdering, qfit, w, a, y, available_covars)
    logitQnAA = predict(qfit, w, a, :link)
    scores = Dict{Int, Float64}()
    wa = [w a]
    aidx = size(wa, 2)
    for i in available_covars
        fit = lreg(wa, y, subset=[i, aidx], offset=logitQnAA)
        r = predict(fit, wa, offset=logitQnAA)
        @devec r[:] = - y .* log(r) .- (1.0 .- y) .* log(1.0 .- r)
        scores[i] = mean(r)
    end
    sort!(collect(keys(scores)), by = x -> scores[x])
end

function order_covars(ordering::PartialCorrOrdering, qfit, w, a, y, available_covars)
    resid = y .- predict(qfit, w, a, :link)
    if available_covars == IntSet(1:size(w, 2))
        scores = vec(abs!(pcor(resid, w, a)))
        return sortperm(scores, rev=true)
    else
        w_available = w[:, collect(available_covars)]
        scores = Dict(collect(zip(available_covars, abs!(pcor(resid, w_available, a)))))
        return sort!(collect(keys(scores)), by = x -> scores[x], rev=true)
    end
end

function order_covars(ordering::HDPSOrdering, qfit, w, a, y, available_covars)
    error()
end

