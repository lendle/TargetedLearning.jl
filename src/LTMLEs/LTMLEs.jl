module LTMLEs

using Logging
if isdefined(Main, :LTMLE_DEBUG) && Main.LTMLE_DEBUG
    @Logging.configure(level=DEBUG)
else
    @Logging.configure(level=OFF)
end

include("DataSets.jl")
include("Regimes.jl")
using Reexport, ..LReg
@reexport using .DataSets, .Regimes
include("G.jl")


end

#=
define
* data
** things like y(1) = 1 -> y (t:t>1) = 1 or whatever
** similar with c
** Subset on time, if meets rule, etc
* dynamic treatments d
* compute g
* compute q

=#
