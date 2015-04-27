using Lexicon, TargetedLearning

fmnames = [("tmles", TargetedLearning.TMLEs),
           ("ctmles", TargetedLearning.CTMLEs),
           #("strategies", TargetedLearning.CTMLEs.Strategies),
           ("common", TargetedLearning.Common),
           ("lreg", TargetedLearning.LReg),
           ("parameters", TargetedLearning.Parameters),
           #("qmodels", TargetedLearning.Qmodels)
           ]

for (fname, mod) in fmnames
	save(joinpath(Pkg.dir("TargetedLearning"), "docs", "api", "$fname.md"), mod, mathjax=true, include_internal = false)
end

nbpath = joinpath(Pkg.dir("TargetedLearning"), "docs", "user-guide", "lalonde_example.ipynb")
mdpath = joinpath(Pkg.dir("TargetedLearning"), "docs", "user-guide", "lalonde_example.md")

run(`ipython nbconvert --to markdown $nbpath --output=$mdpath`)
