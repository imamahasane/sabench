.PHONY: install install-dev install-audio test lint typecheck demo validate figures clean \
        prepare-audio extract-lld extract-embeddings e0 e1 e2 e3 e4 prepare final

PYTHON ?= python3
DATASET ?= adress
FEATURESET ?= eGeMAPSv02
MODEL ?= wavlm

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e . -r requirements-dev.txt

install-audio:
	$(PYTHON) -m pip install -e . -r requirements-extra.txt

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests

typecheck:
	mypy src

demo:
	sabench demo

validate:
	sabench validate --dataset all

prepare-audio:
	sabench prepare-audio $(DATASET)

extract-lld:
	sabench extract-lld $(DATASET) --featureset $(FEATURESET)

extract-embeddings:
	sabench extract-embeddings $(DATASET) --model $(MODEL)

e0:
	sabench replicate $(DATASET) --featureset $(FEATURESET)

e1:
	sabench null $(DATASET) --featureset $(FEATURESET)

e2:
	sabench uncertainty $(DATASET) --featureset $(FEATURESET)

e3:
	sabench probes $(DATASET) --model $(MODEL)

e4:
	sabench segment-channel $(DATASET) --featureset $(FEATURESET)

figures:
	sabench figures

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info src/*.egg-info build dist htmlcov .coverage

# Full study: build every input from ../Data, then regenerate every table and figure.
prepare:
	bash scripts/prepare_all.sh

final:
	bash scripts/run_final.sh
