.PHONY: install test smoke generate leakage train evaluate app

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -c "import cod_sentinel; print(cod_sentinel.__version__)"

generate:
	$(PYTHON) -m cod_sentinel.generator

leakage:
	$(PYTHON) -m cod_sentinel.leakage

train:
	$(PYTHON) -c "from cod_sentinel.models import main; main()"

evaluate:
	$(PYTHON) -c "from cod_sentinel.evaluation import main; main()"

app:
	@echo "$@ is intentionally unavailable until its implementation milestone."
	@exit 2
