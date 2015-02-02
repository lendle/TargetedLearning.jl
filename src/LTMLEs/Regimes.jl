module Regimes

export Regime, setintervention!

using ..DataSets


immutable Regime
    values::Dict{Anode,Union(Float64, Vector{Float64})}    
end

function setintervention!(ds::DataSet, regime::Regime)
    dsanodes = Anodes(ds.nodes)
    for node in dsanodes
        ds.data[:, indexes(node)] = regime.values[node]
    end
    ds
end

end
