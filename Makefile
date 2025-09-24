.PHONY: help install install-extras preview preview-ci ac-bode step-fig mc-fig opamp-stability pt1000 examples

VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
TEMP?=100
N?=1000
SIGMA?=0.01

help:
	@echo "Targets:"
	@echo "  make install     # create venv and install package in editable mode"
	@echo "  make install-extras  # install matplotlib for plotting"
	@echo "  make preview     # run circuit preview example (uses venv if present)"
	@echo "  make preview-ci  # run preview without venv"
	@echo "  make ac-bode     # run AC Bode example (requires ngspice + matplotlib)"
	@echo "  make step-fig    # run STEP grid example (requires ngspice + matplotlib)"
	@echo "  make mc-fig      # run Monte Carlo figures (requires ngspice + matplotlib)"
	@echo "  make opamp-stability # run op-amp Bode example (requires ngspice + matplotlib)"
	@echo "  make pt1000      # run PT1000 Monte Carlo (TEMP=$(TEMP) N=$(N) SIGMA=$(SIGMA))"
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
		python3 -m examples.circuit_preview; \
	fi

preview-ci:
	python3 -m examples.circuit_preview

ac-bode:
	python3 -m examples.ac_bode

step-fig:
	python3 -m examples.step_sweep_fig

mc-fig:
	python3 -m examples.monte_carlo_fig

opamp-stability:
	python3 -m examples.opamp_stability

pt1000:
	python3 -m examples.pt1000_mc --temp $(TEMP) --n $(N) --sigma $(SIGMA) --workers 4

examples: ac-bode step-fig mc-fig opamp-stability
