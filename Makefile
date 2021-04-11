COMMIT_HASH := $(shell eval git rev-parse HEAD)

cython:
	python setup.py build_ext --inplace --force

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/*/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:

doc: render-notebooks
	python docs/scripts/index.py
	mkdocs build

livedoc: doc
	mkdocs serve --dirtyreload

.PHONY: bench
bench:
	asv run ${COMMIT_HASH} --config benchmarks/asv.conf.json --steps 1
	asv run master --config benchmarks/asv.conf.json --steps 1
	asv compare the-merge ${COMMIT_HASH} --config benchmarks/asv.conf.json