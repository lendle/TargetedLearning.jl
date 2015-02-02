module DataSets

using Docile
@docstrings

export Node, makenode, Lnode, Anode, Cnode, Ynode, ACnode, LYnode,
       Nodes, Lnodes, Anodes, Cnodes, ACnodes, Ynodes, LYnodes,
       indexes, parents, 

       DataSet,
       nondeterministic, uncensored


@doc "A object of type `Node` holds the index(es) into the data matrix associated with 
a particular node" ->
abstract Node

@doc "Helper function to create an :L, :A, :C, or :Y node where T represents the node type." ->
function makenode(T::Symbol, idx)
    T == :L? Lnode(idx) :
    T == :A? Anode(idx) :
    T == :C? Cnode(idx) :
    T == :Y? Ynode(idx) :
        error()
end
makenode(T, idx) = makenode(symbol(T), idx)

@doc """
Represents a  baseline and time varying covariate node. A single Lnode can represent more than one 
column of the data matrix at once by passing multiple indexes or a vector of indexes to `Lnode`
""" ->
immutable Lnode <: Node
    idx::Vector{Int}
end
Lnode(i::Int...) = Lnode([i...])
==(a::Lnode, b::Lnode) = a.idx == b.idx
function Base.show(io::IO, node::Lnode)
    print(io, "Lnode")
    Base.show_delim_array(io, node.idx, '(', ',', ')', true)
end

@doc "Represents a treatment node" -> 
immutable Anode <: Node
    idx::Int
end

@doc "Represents a censoring node" -> 
immutable Cnode <: Node
    idx::Int
end

@doc "Represents an outcome node" -> 
immutable Ynode <: Node
    idx::Int
end

typealias LYnode Union(Lnode, Ynode)
typealias ACnode Union(Anode, Cnode)


typealias Nodes Vector{Node}

@doc "Returns only `Lnode`s from a vector of `Node`s" -> 
Lnodes(nodes::Nodes) = filter(nd -> isa(nd, Lnode), nodes)
@doc "Returns only `Anode`s from a vector of `Node`s" -> 
Anodes(nodes::Nodes) = filter(nd -> isa(nd, Anode), nodes)
@doc "Returns only `Cnode`s from a vector of `Node`s" -> 
Cnodes(nodes::Nodes) = filter(nd -> isa(nd, Cnode), nodes)
@doc "Returns only `Anode`s and `Cnodes` from a vector of `Node`s" -> 
ACnodes(nodes::Nodes) = filter(nd -> isa(nd, ACnode), nodes)
@doc "Returns only `Ynode`s from a vector of `Node`s" -> 
Ynodes(nodes::Nodes) = filter(nd -> isa(nd, Ynode), nodes)

# returns a Nodes where each node is the first L or Y node after one or more A or C nodes
@doc "Returns a vector of the first L or Y node after one or more A or C nodes from a vector of `Node`s" -> 
function LYnodes(nodes::Nodes)
    lynodes = Node[]
    prev_node_ac = false #was previous node an AC node?
    for node in nodes
        if prev_node_ac && isa(node, LYnode)
            push!(lynodes, node)
        end
        prev_node_ac = isa(node, ACnode)
    end
    lynodes
end

@doc "Returns all nodes in coming before `node` in a vector of `Node`s" ->
function parents(nodes::Nodes, node::Node)
    nodeidx = findnext(nd -> nd == node, nodes, 1)
    nodeidx == 0 && error("node $node not in nodes")
    nodes[1:nodeidx-1]
end

@doc* "Retruns an integer vector of indexes corresponding to all `Node`s in `nodes`." ->
indexes(node::Node) = node.idx
indexes(nodes::Nodes) = sort!(vcat(map(indexes, nodes)...))


type DataSet
    data::Matrix{Float64}
    nodes::Nodes
    survivalOutcome::Bool
end
Base.copy(ds::DataSet) = DataSet(copy(ds.data), copy(ds.nodes), ds.survivalOutcome)


@doc "Returns a logical vector with true for each observation that is nondeterministic BEFORE
the node of interest and false otherwise." ->
function nondeterministic(ds::DataSet, node::Node)
    n = size(ds.data, 1)
    !ds.survivalOutcome && return fill(true, n)
    past_ynodes = Ynodes(parents(ds.nodes, node))
    isempty(past_ynodes) && return fill(true, n)
    
    prev_ynode_idx = indexes(past_ynodes[end])
    
    y = ds.data[:, prev_ynode_idx]
    return y .!= 1.0
end

@doc "Returns a logical vector with true for each observation that is uncensored BEFORE
the node of interest and false otherwise." ->
function uncensored(ds::DataSet, node::Node)
    n = size(ds.data, 1)
    past_cnodes = Cnodes(parents(ds.nodes, node))
    isempty(past_cnodes) && return fill(true, n)
    
    prev_cnode_idx = indexes(past_cnodes[end])
    
    c = ds.data[:, prev_cnode_idx]
    return c .!= 1.0
end

end
