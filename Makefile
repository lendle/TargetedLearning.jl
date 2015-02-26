.PHONY: docs test lint

docs:
	julia docs/build.jl
	mkdocs build --clean

test:
	julia -e 'Pkg.test("TargetedLearning")'

lint:
	julia -e 'using Lint; lintpkg("TargetedLearning")'
