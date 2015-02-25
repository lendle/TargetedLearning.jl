.PHONY: docs

docs:
	julia docs/build.jl
	mkdocs build --clean