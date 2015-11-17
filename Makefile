.PHONY: docs test lint test-covarage

docs:
	julia docs/build.jl
	mkdocs build --clean

test:
	julia --color=yes test/runtests.jl

test-coverage:
	julia --color=yes -e 'Pkg.test("TargetedLearning", coverage=true)'

lint:
	julia --color=yes -e 'using Lint; lintpkg("TargetedLearning")'
