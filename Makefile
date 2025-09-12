.PHONY: help install preview preview-ci

VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

help:
	@echo "Targets:"
	@echo "  make install     # create venv and install package in editable mode"
	@echo "  make preview     # run circuit preview example (uses venv if present)"
	@echo "  make preview-ci  # run preview with PYTHONPATH=src (no venv)"

install:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e .

preview:
	@if [ -x "$(PY)" ]; then \
		"$(PY)" -m examples.circuit_preview; \
	else \
		PYTHONPATH=src python3 -m examples.circuit_preview; \
	fi

preview-ci:
	PYTHONPATH=src python3 -m examples.circuit_preview
