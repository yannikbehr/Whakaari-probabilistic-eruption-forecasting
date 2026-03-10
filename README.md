## Probabilistic eruption forecasting for Whakaari/White Island using Bayesian Networks

This is the source code accompanying the peer-reviewed [publication in Geophysical Research Letters](https://doi.org/10.1029/2024GL112029) on probabilistic eruption forecasting for Whakaari/White Island using Bayesian Networks.

Maintainer: [Yannik Behr](mailto:y.behr@gns.cri.nz)

## Installation

### Dependencies
The python package has an optional dependency on the SMILE library for Bayesian Network learning and inference. The library is written and maintained by [BayesFusion](https://www.bayesfusion.com/) and is free for academic users.

If SMILE is not installed, the package will use [pgmpy](https://pgmpy.org/) instead.

### Download the source code using git

```
git clone https://github.com/yannikbehr/whakaari_eruption_forecasting.git
cd whakaari_eruption_forecasting
```

### Installing the package and its dependencies in a conda environment 
```
conda create whakaaribn python=3.11 
conda activate whakaaribn
pip install -e .
```
### Installing the PYSMILE library
```
pip install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile
```
Next obtain a license from [BayesFusion](https://www.bayesfusion.com/).
Academic users can obtain a free license, all other users can get a 30-day evaluation license
here: (https://download.bayesfusion.com/files.html?category=Business).
Once you have unzipped the package with license files, run the following command to copy the license key to the correct place:

```
cp pysmile_license.py $(python -c "import site; print(site.getsitepackages()[0]))
```

## Command-line interface (CLI)

The `whakaaribn` command provides access to the Snakemake workflows used for
benchmarking and scheduled forecasting runs.

### Show help
```
whakaaribn -h
```

### Run the benchmark workflow

To reproduce the results from the publication you can run the benchmark workflow.
The benchmark subcommand requires a target directory where results will be
written.

```
whakaaribn benchmark --directory /path/to/output
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--backend {pgmpy,smile}` | `pgmpy` | Bayesian Network backend to use |
| `--cores N` | `1` | Number of cores to pass to Snakemake |
| `--clean` | `false` | Delete all output files before running |

> **Note:** Generating plots requires [Google Chrome](https://www.google.com/chrome/) to be installed
> (used by [kaleido](https://github.com/plotly/Kaleido) to export Plotly figures to PNG).
> If Chrome is not available on your system, you can use the `scripts/build_and_benchmark.sh`
> script instead, which runs the benchmark inside a Docker container that has Chrome pre-installed:
>
> ```
> scripts/build_and_benchmark.sh --build --data /path/to/output
> ```
>
> To include SMILE support in the Docker image, place your `pysmile_license.py` file in the
> project root before building. The Dockerfile will automatically detect it and install pysmile.
> Without the license file, the image is built without SMILE and falls back to pgmpy.

### Run the scheduled daily workflow (daemon)

Run the workflow once immediately, then every day at 13:00 UTC. The default
directory is `/opt/data`.

```
whakaaribn daemon
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--directory /path/to/data` | `/opt/data` | Directory to run the workflow in |
| `--backend {pgmpy,smile}` | `pgmpy` | Bayesian Network backend to use |
| `--cores N` | `1` | Number of cores to pass to Snakemake |
| `--clean` | `false` | Delete all output files before running |
