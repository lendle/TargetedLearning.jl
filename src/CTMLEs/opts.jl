immutable CTMLEOpts
    searchstrategy::SearchStrategy
    ginitidx::Vector{Int}
end

CTMLEOpts(;searchstrategy::SearchStrategy=ForwardStepwise(),
          ginitidx=[1]) = CTMLEOpts(deepcopy(searchstrategy), collect(ginitidx))
