test:
	python -m pytest -q

build:
	python -m build

lint:
	ruff check src tests vpipe
