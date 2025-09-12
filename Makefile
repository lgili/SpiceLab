.PHONY: help install install-extras preview preview-ci ac-bode step-fig mc-fig opamp-stability examples

VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

help:
	@echo "Targets:"
	@echo "  make install     # create venv and install package in editable mode"
	@echo "  make install-extras  # install matplotlib for plotting"
	@echo "  make preview     # run circuit preview example (uses venv if present)"
	@echo "  make preview-ci  # run preview with PYTHONPATH=src (no venv)"
	@echo "  make ac-bode     # run AC Bode example (requires ngspice + matplotlib)"
	@echo "  make step-fig    # run STEP grid example (requires ngspice + matplotlib)"
	@echo "  make mc-fig      # run Monte Carlo figures (requires ngspice + matplotlib)"
	@echo "  make opamp-stability # run op-amp Bode example (requires ngspice + matplotlib)"
	@echo "  make examples    # run all figure examples"

install:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e .

install-extras:
	@if [ -x "$(PIP)" ]; then \
		"$(PIP)" install matplotlib; \
	else \
		pip install matplotlib; \
	fi

preview:
	@if [ -x "$(PY)" ]; then \
		"$(PY)" -m examples.circuit_preview; \
	else \
		PYTHONPATH=src python3 -m examples.circuit_preview; \
	fi

preview-ci:
	PYTHONPATH=src python3 -m examples.circuit_preview

ac-bode:
	PYTHONPATH=src python3 -m examples.ac_bode

step-fig:
	PYTHONPATH=src python3 -m examples.step_sweep_fig

mc-fig:
	PYTHONPATH=src python3 -m examples.monte_carlo_fig

opamp-stability:
	PYTHONPATH=src python3 -m examples.opamp_stability

examples: ac-bode step-fig mc-fig opamp-stability
