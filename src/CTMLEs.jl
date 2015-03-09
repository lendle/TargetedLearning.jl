module CTMLEs


function ctmle(logitQnA1, logitQnA0, w, a, y;
               cvplan = StratifiedRandomSub(zip(a, y), iround(0.9 * length(y)), 1))

    if isa(cvplan, CrossValGenerator)
        cvplan = collect(cvplan)
    end

    qmodels = CtmleQmodels[] #setup
    steps = 0
    while !done
        steps +=1
        vallosses = Float64[]
        for qmodel in qmodels
            addcovar(qmodel, w, a, y)
            push!(vallosses, val_loss(qmodel, a, y))
        end
        valloss = mean(valloss)
        done = check_if_done(vallosses, history)

    end

    fullqmodel #setup with no val

    for i in 1:steps
        addcovar(fullqmodel)
    end

    compute_est_with(fullqmodel)

end



end # module


