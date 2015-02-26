using Lexicon, TargetedLearning

fmnames = [("ctmles", TargetedLearning.CTMLEs),
           ("strategies", TargetedLearning.CTMLEs.Strategies),
           ("lreg", TargetedLearning.LReg)]

for (fname, mod) in fmnames
	save("docs/api/$fname.md", mod, mathjax=true)
end
