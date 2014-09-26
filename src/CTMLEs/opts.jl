immutable CTMLEOpts
    searchstrategy::SearchStrategy
end

CTMLEOpts(;searchstrategy::SearchStrategy=ForwardStepwise()) = CTMLEOpts(deepcopy(searchstrategy))
