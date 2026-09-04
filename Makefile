.PHONY: install test smoke pipeline generate leakage train freeze evaluate app

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -c "import cod_sentinel; print(cod_sentinel.__version__)"

pipeline:
	$(MAKE) generate PYTHON=$(PYTHON)
	$(MAKE) leakage PYTHON=$(PYTHON)
	$(MAKE) train PYTHON=$(PYTHON)
	$(MAKE) freeze PYTHON=$(PYTHON)
	$(MAKE) evaluate PYTHON=$(PYTHON)

generate:
	$(PYTHON) -m cod_sentinel.generator

leakage:
	$(PYTHON) -m cod_sentinel.leakage

train:
	$(PYTHON) -c "from cod_sentinel.models import main; main()"

freeze:
	$(PYTHON) -c "from cod_sentinel.evaluation import freeze_main; freeze_main()"

evaluate:
	$(PYTHON) -c "from cod_sentinel.evaluation import main; main()"

app:
	$(PYTHON) -m streamlit run app.py
