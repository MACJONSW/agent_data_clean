PYTHON ?= python
ROOT_DIR := $(shell pwd)

.PHONY: help test validate-example run-example smoke docker-build clean pycompile

help:
	@echo "Targets:"
	@echo "  test             Run unit tests"
	@echo "  pycompile        Run syntax checks"
	@echo "  validate-example Validate dev config"
	@echo "  run-example      Run example pipeline"
	@echo "  smoke            Run tests + config validation"
	@echo "  docker-build     Build container image"
	@echo "  clean            Remove generated caches and outputs"

pycompile:
	$(PYTHON) -m py_compile run_agent_posttrain_pipeline.py
	$(PYTHON) -m py_compile agent_posttrain_pipeline/*.py monitoring/*.py scripts/validate_config.py scripts/check_monitoring_artifacts.py

test:
	$(PYTHON) -m unittest discover -s tests/agent_posttrain_pipeline -p 'test_*.py'

validate-example:
	$(PYTHON) scripts/validate_config.py --config configs/dev.yaml

run-example:
	bash scripts/run_example.sh

smoke: pycompile test validate-example

docker-build:
	docker build -t agent-posttrain-pipeline:latest .

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf outputs
