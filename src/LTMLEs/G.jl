type Gfit
    fits::Dict{ACnode, AbstractLR}
end

function fitg(ds) 
    acnodes = ACnodes(ds.nodes)
    fits = Dict{ACnode, AbstractLR}()
    for node in acnodes
        @info "Fitting model for $node"
        sub = nondeterministic(ds, node) & uncensored(ds, node)

        parnodes = parents(ds.nodes, node)
        if ds.survivalOutcome
            predictornodes = filter(node -> isa(node, Union(Anode, Lnode)), parnodes)
        else
            predictornodes = filter(node -> !isa(node, Cnode), parnodes)
        end
        
        idx = indexes(predictornodes)
        x = ds.data[sub, idx]
        y = ds.data[sub, indexes(node)]
        fit = LReg.lreg_bfgs(x, y)
        coef = zeros(size(ds.data, 2))
        coef[idx] = fit.beta
        fits[node] = LReg.LR(coef)
    end
    Gfit(fits)
end