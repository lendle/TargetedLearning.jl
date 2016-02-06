abstract OrderingStrategy
type LogisticOrdering <: OrderingStrategy end
type PartialCorrOrdering <: OrderingStrategy end
type HDPSOrdering <: OrderingStrategy end
type ManualOrdering <: OrderingStrategy
  indexes::Vector{Int}
  ManualOrdering(indexes) = new(copy(indexes))
end

"""
Defines how the next covariate added to g is chosen

* `ForwardStepwise` - default, tries each unused covariate to find the one that helps the most. Slow, O(p^2) time for p covariates.
* `PreOrdered` - Orders covariates once then adds them in that order. Takes O(p) time for p covariates. Specify with `PreOrdered(ordering)` where `ordering` is
    * `LogisticOrdering()` - ranks covariates based on reduction in log likelihood with intital Q as an offset if added one at a time. Basically the first step of `ForwardStepwise`
    * `PartialCorrOrdering()` - ranks covariates based on partial correlation of each covariate and the residual (y - predicted y from initial Qbar) given a
    * `HDPSOrdering()` - ranks covariates based on bias reduction potential as in the hdps procedure. This ignores intial Qbar.
    * `ManualOrdering(ord)` - orders covars based on specified `indexes`, a `Vector{Int}`, of column indexes of W, not including 1, the intercept.
"""
abstract SearchStrategy
type ForwardStepwise <: SearchStrategy end
type PreOrdered <: SearchStrategy
    ordering::OrderingStrategy
    k_at_once::Int
    covar_order::Vector{Int}
    PreOrdered(ordering::OrderingStrategy, k_at_once=1) = new(ordering, k_at_once, Array(Int, 0))
end
Base.show(io::IO, strat::PreOrdered) = print(io, "Preordered($(strat.ordering))")

#strategies may store some state and we need to clear it out sometimes
init!(strat::SearchStrategy) = strat
function init!(strat::PreOrdered)
    strat.covar_order = Int[]
    strat
end

# added covar(s) are added to chunk.used_covars and deleted from chunk.unused_covars
# q is fluctuated
#return best_risk, gfit, new_fluc
function add_covar!(::ForwardStepwise, chunk::Qchunk)
    q=chunk.q
    W=chunk.W
    A=chunk.A
    Y=chunk.Y
    prev_risk=chunk.risk
    @debug("unused_covars: $(chunk.unused_covars)")
    @debug("used_covars: $(chunk.used_covars)")

    next_covar_risk = fill(Inf, maximum(chunk.unused_covars))
    g_fluc_dict = Dict{IntSet, Tuple{LR, Fluctuation}}()

    #Remove most recent fluctuation (and save for later use)
    #So that we can add one covar to the current fluctuation to see if it works
    fluc_now = defluctuate!(q)

    #try adding each covariate to the current fluctuation
    for j in chunk.unused_covars
        #estimate g with additional covariate j
        #println("Vars: $(union(chunk.used_covars, IntSet(j)))")
        current_covars = union(chunk.used_covars, IntSet(j))
        gfit = give_me_a_g(chunk.gmachine, subset=collect(current_covars))
        #fluctuate and get risk
        fluc = computefluc(q, chunk.param, bound!(predict(gfit, W), chunk.gbounds), A, Y)
        fluctuate!(q, fluc)
        next_covar_risk[j] = risk(q, A, Y, chunk.param, chunk.penalize)
        if isnan(next_covar_risk[j])
            @warn("Risk is NaN. used covars: $chunk.used_covars")
        end
        #remove newest fluctuation
        defluctuate!(q)
        g_fluc_dict[current_covars] = (gfit, fluc)
    end
    best_risk, best_j = findmin(next_covar_risk)

    if best_risk < prev_risk
        @debug("Adding to old fluc, best_j:$best_j")
        new_fluc = false
        #adding one covariate to current fluc does better, then use that fluctuation
        push!(chunk.used_covars, best_j)
        @debug("used_covars after adding best_j: $chunk.used_covars")
        delete!(chunk.unused_covars, best_j)
        gfit, fluc = g_fluc_dict[chunk.used_covars]
        fluctuate!(q, fluc)
    else
        new_fluc = true
        #otherwise, keep current fluctuation find best new fluctuation adding a
        #single covariate
        fluctuate!(q, fluc_now) #reuse previous best fluctuation
        next_covar_risk = fill(Inf, maximum(chunk.unused_covars))
        for j in chunk.unused_covars
            current_covars = union(chunk.used_covars, IntSet(j))
            gfit = g_fluc_dict[current_covars][1]
            fluc = computefluc(q, chunk.param, bound!(predict(gfit, W), chunk.gbounds), A, Y)
            fluctuate!(q, fluc)
            next_covar_risk[j] = risk(q, A, Y, chunk.param, chunk.penalize)
            if isnan(next_covar_risk[j])
                @warn("Risk is NaN. used covars: $chunk.used_covars")
            end
            defluctuate!(q)
            g_fluc_dict[current_covars] = (gfit, fluc)
        end
        @debug("next_covar_risk: $next_covar_risk")
        best_risk, best_j = findmin(next_covar_risk)
        if !chunk.penalize && best_risk > prev_risk
            @warn("Best risk is worse than previous best risk. This doesn't make sense. Continuing anyway...")
        end
        @debug("best_j: $best_j, unused_covars: $chunk.unused_covars")
        push!(chunk.used_covars, best_j)
        delete!(chunk.unused_covars, best_j)
        gfit, fluc = g_fluc_dict[chunk.used_covars]
        fluctuate!(q, fluc)
    end
    best_risk, gfit, new_fluc
end

# added covar(s) are added to chunk.used_covars and deleted from chunk.unused_covars
# q is fluctuated
#return best_risk, gfit, new_fluc
function add_covar!(strategy::PreOrdered, chunk::Qchunk)
    q=chunk.q
    W=chunk.W
    A=chunk.A
    Y=chunk.Y
    prev_risk=chunk.risk

    if isempty(strategy.covar_order)
        append!(strategy.covar_order, order_covars(strategy.ordering, chunk))
    end

    ordered_unused_covars = filter(x -> x ∉ chunk.used_covars, strategy.covar_order)
    next_covars = length(ordered_unused_covars) >= strategy.k_at_once ?
        ordered_unused_covars[1:strategy.k_at_once] :
        ordered_unused_covars


    fluc_now = defluctuate!(q)

    union!(chunk.used_covars, next_covars)
    setdiff!(chunk.unused_covars, next_covars)

    g_fit = give_me_a_g(chunk.gmachine, subset=collect(chunk.used_covars))
    gn1 = bound!(predict(g_fit, W), chunk.gbounds)

    fluc = computefluc(q, chunk.param, gn1, A, Y)
    fluctuate!(q, fluc)
    new_risk = risk(q, A, Y, chunk.param, chunk.penalize)

    if new_risk < prev_risk
        return new_risk, g_fit, false
    end

    defluctuate!(q)
    fluctuate!(q, fluc_now)
    fluctuate!(q, chunk.param, gn1, A, Y)
    return risk(q, A, Y, chunk.param, chunk.penalize), g_fit, true
end

function order_covars(ordering::LogisticOrdering, chunk)
    q=chunk.q
    W=chunk.W
    A=chunk.A
    Y=chunk.Y
    available_covars=chunk.unused_covars
    logitQnAA = linpred(q, A)
    scores = Dict{Int, Float64}()
    for i in available_covars
        g_fit = give_me_a_g(chunk.gmachine, subset=[i])
        gn1 = bound!(predict(g_fit, W), chunk.gbounds)
        fluc = computefluc(q, chunk.param, gn1, A, Y)
        fluctuate!(q, fluc)
        scores[i] = risk(q, A, Y, chunk.param, chunk.penalize)
        defluctuate!(q)
    end
    sort!(collect(keys(scores)), by = x -> scores[x])
end

function order_covars(ordering::PartialCorrOrdering, chunk)
    q=chunk.q
    W=chunk.W
    A=chunk.A
    Y=chunk.Y
    available_covars=chunk.unused_covars
    resid = Y .- predict(q, A)
    W_available = W[:, collect(available_covars)]
    scores = Dict(collect(zip(available_covars, abs(pcor(resid, W_available, A)))))
    return sort!(collect(keys(scores)), by = x -> scores[x], rev=true)
end

function order_covars(ordering::HDPSOrdering, chunk)
    error()
end

function order_covars(ordering::ManualOrdering, chunk)
    available_covars=chunk.unused_covars
    collect(available_covars) == sort(ordering.indexes) ||
      error("ManualOrdering indexes do not match available covariates")
    ordering.indexes
end
