.PHONY: docs test lint test-covarage

docs:
	julia docs/build.jl
	mkdocs build --clean

test:
	julia --color -e 'Pkg.test("TargetedLearning")'

test-coverage:
	julia --color -e 'Pkg.test("TargetedLearning", coverage=true)'

lint:
	julia --color -e 'using Lint; lintpkg("TargetedLearning")'
