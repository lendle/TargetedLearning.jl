module TestTargetedLearning

using FactCheck
FactCheck.setstyle(:compact)

is_a(T) = x -> isa(x, T)

include("LReg.jl")
# # include("ctmle.jl")
include("automaticics.jl")
include("Qmodels.jl")
include("Parameters.jl")
include("TMLEs.jl")

FactCheck.exitstatus()
end
