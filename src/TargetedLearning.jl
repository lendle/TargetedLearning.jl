module TargetedLearning

using Docile
@document

using Reexport

include("Common.jl")
include("LReg.jl")
include(joinpath("CTMLEs", "CTMLEs.jl"))
include("TMLEs.jl")

end
