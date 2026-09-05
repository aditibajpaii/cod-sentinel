.PHONY: install test smoke pipeline generate leakage train freeze evaluate plots ledger-sample app install-agent agent-demo webhook

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-agent:
	$(PYTHON) -m pip install -e ".[dev,agent]"

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

plots:
	$(PYTHON) -c "from cod_sentinel.plots import main; main()"

ledger-sample:
	$(PYTHON) -c "from cod_sentinel.ledger import main; main()"

app:
	$(PYTHON) -m streamlit run app.py

webhook:
	$(PYTHON) -c "from cod_sentinel.orchestrator.webhook import main; main()"

agent-demo:
	@test -n "$(ORDER_ID)" || (echo "Set ORDER_ID=..." && exit 1)
	@test -n "$(ADDRESS)" || (echo "Set ADDRESS=..." && exit 1)
	@test -n "$(PHONE)" || (echo "Set PHONE=..." && exit 1)
	$(PYTHON) -m cod_sentinel.orchestrator --order-id "$(ORDER_ID)" --address "$(ADDRESS)" --phone "$(PHONE)"
