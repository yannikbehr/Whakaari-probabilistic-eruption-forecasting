# Copilot Instructions for Whakaari Probabilistic Eruption Forecasting

## Project Overview

This repository contains the source code for probabilistic eruption forecasting for Whakaari/White Island (New Zealand) using Bayesian Networks. It accompanies a manuscript on multi-sensor eruption forecasting. The core monitoring parameters include earthquake rates (Eqr), CO₂, RSAM, SO₂, and H₂S emissions.

The main Python package is `whakaaribn`.

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- A conda environment is recommended (see `README.md`)

### Install dependencies

```bash
conda env create -f environment.yml
conda activate whakaaribn
pip install -e ".[dev]"
```

### PYSMILE library (required for Bayesian Network inference)

```bash
pip install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile
# Then copy your BayesFusion license file:
cp pysmile_license.py $(python -c "import site; print(site.getsitepackages()[0])")
```

## Running Tests

Tests use `pytest` and are located in `whakaaribn/tests/`.

```bash
pytest --pyargs whakaaribn.tests
```

Or from the repository root:

```bash
pytest
```

## Linting

This project uses **Ruff** for linting. Unused imports (`F401`) are flagged but not auto-fixed.

```bash
ruff check .
ruff format .
```

## Project Structure

```
whakaaribn/               # Main package
├── api.py                # FastAPI server (whakaari_api entry point)
├── bayesnet.py           # Wrapper around the SMILE/pysmile Bayesian Network library
├── cli.py                # CLI entry points: bayes_daemon, whakaaribn
├── forecast.py           # Forecasting logic and scheduling
├── model.py              # WhakaariModel: Bayesian Network learning via pgmpy
├── grid_search.py        # Hyperparameter grid search
├── assimilate.py         # Data assimilation utilities
├── visualize.py          # Plotting and visualization
├── util.py               # Core utilities (Discretizer, BinData, imputers, etc.)
├── data/                 # Bundled data files (JSON configs, eruption catalogue CSV)
└── tests/                # pytest test suite
examples/
└── Whakaari_BN.ipynb     # Jupyter notebook reproducing manuscript figures
```

## Key Architectural Components

- **`model.py` / `WhakaariModel`**: Learns the Bayesian Network structure and parameters from data using `pgmpy`'s `ExpectationMaximization`.
- **`bayesnet.py`**: Wraps the `pysmile` (SMILE) library for probabilistic inference on the trained network.
- **`api.py`**: FastAPI server exposing `/forecast` and `/labels` endpoints, integrates with the Tonik data framework.
- **`cli.py`**: Snakemake-based workflow orchestration; `bayes_daemon` runs forecasts on a schedule (daily at 13:00 UTC).
- **`forecast.py`**: Scheduling and execution logic for daily eruption probability forecasts.
- **`util.py`**: Data discretization, bin handling, and imputation utilities used throughout the pipeline.

## Code Style Guidelines

- Follow **PEP 8** and use type hints where appropriate.
- Use `ruff` for linting and formatting — do not auto-fix `F401` (unused imports).
- Keep public functions and classes documented with docstrings (validated by `test_docstrings.py`).
- Tests are co-located under `whakaaribn/tests/` and use `pytest` with fixtures defined in `conftest.py`.
- When adding new monitoring parameters or model features, update the data files in `whakaaribn/data/` and the corresponding discretization logic in `util.py`.

## CLI Entry Points

| Command | Module | Description |
|---------|--------|-------------|
| `whakaaribn` | `whakaaribn.cli:main` | Main CLI for running forecasts and workflows |
| `bayes_daemon` | `whakaaribn.cli:main` | Daemon mode for scheduled daily forecasting |
| `whakaari_api` | `whakaaribn.api:main` | Start the FastAPI forecast server |

## Deployment

The project is deployed as two Docker services (see `gessp-operational-stack.yml`):
- `bayes_daemon`: runs the Snakemake workflow daily
- `whakaari_api`: serves the FastAPI forecast API on port 8049

CI/CD is configured via `.gitlab-ci.yml` with build → test → deploy stages.
