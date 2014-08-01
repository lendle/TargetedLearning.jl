#Qmodel keeps track of initial Q estimate (by logistic reg) and
#sequence of g estimates and fluctuations

type Qmodel
    Qinit::SSLR
    gseq::Vector{AbstractLR}
    flucseq::Vector{LR}
end

Qmodel(qinit::AbstractLR) = Qmodel(qinit, AbstractLR[], LR[])

function Base.show(io::IO, obj::Qmodel)
    idxs = [IntSet(), map(x-> x.idx, obj.gseq)]
    print(io, "Initial Q fit on covars:")
    show(io, obj.Qinit.idx)
    println(io,"")
    for i in 2:length(idxs)
        added = setdiff(idxs[i], idxs[i-1])
        println(io, "$(i-1)th fluctuation, covars added: $(collect(added))")
    end
    println(io, "$(length(obj.gseq[end].idx)) of $(length(obj.gseq[end].beta)) covariates in fluctuations.")
end

function predict!(q::Qmodel, r, w, a, kind)
    #predicts Qbar by using the initial Q estimate then looping through fluctuations
    #result is stored in r
    @assert length(q.gseq) == length(q.flucseq)
    n = length(a)
    k = length(q.gseq)
    r = predict(q.Qinit, [w a], :link)  #get initial Q prediction
    if k > 0
        offset = similar(r)
    end
    for i in 1:k  #Loop through fluctuations, updateign prediction
        copy!(offset, r)
        h = predict(q.gseq[i], w, :prob)
        map1!(Gatoh(), h, a)
        predict!(q.flucseq[i], r, reshape(h, length(h), 1), :link, offset=offset)
    end
    if kind==:prob
        map1!(LogisticFun(), r)
    end
    r
end

predict(q::Qmodel, w, a, kind) = predict!(q, similar(a), w, a, kind)

function fluctuate!(q::Qmodel, g::AbstractLR, w, a, y)
    #adds an estimate of g and fluctuation to q
    offset = predict(q, w, a, :link)
    h = predict(g, w, :prob)
    map1!(Gatoh(), h, a)
    h = reshape(h, length(h), 1)
    fluc = lreg(h, y, offset)
    if isnan(fluc.beta[1])
        f = open("asdf", "w")
        serialize(f, (h, y, offset))
        close(f)
    end
    push!(q.gseq, g)
    push!(q.flucseq, fluc)
end

function defluctuate!(q::Qmodel)
    #removes last fluctuation and g
    @assert length(q.gseq) == length(q.flucseq) > 0
    g = pop!(q.gseq)
    (g, pop!(q.flucseq))
end

function refluctuate!(q::Qmodel, g::AbstractLR, fluc::LR)
    push!(q.gseq, g)
    push!(q.flucseq, fluc)
end

function risk(q::Qmodel, w, a, y; pen=true)
    #calculate penalized log likelihood as RSS + tau(=1 here?) * var(ic)
    # need to implement log likelihood
    # note: not scaled by n. ask Susan.

    loss = sum(LReg.Loss(), y, predict(q, w, a, :link))

    if pen
        n = length(y)
        h = predict(q.gseq[end], w, :prob)
        map1!(Gatoh(), h, a)

        QnA1 = predict(q, w, ones(n), :prob)
        QnA0 = predict(q, w, zeros(n), :prob)
        QnAA = ifelse(a.==1, QnA1, QnA0)

        resid = y .- QnAA

        loss += var(h .* resid .+ QnA1 .- QnA0)
    end
    loss
end
