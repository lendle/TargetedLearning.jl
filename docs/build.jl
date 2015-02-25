using Lexicon, TargetedLearning 

fmnames = [("ctmles", TargetedLearning.CTMLEs), 
   		   ("strategies", TargetedLearning.CTMLEs.Strategies)] 

for (fname, mod) in fmnames 
	save("docs/api/$fname.md", mod)
end