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
written. By default it writes results to `/tmp`. 

```
whakaaribn benchmark --directory /path/to/output
```

### Run the scheduled daily workflow (daemon)

Run the workflow once immediately, then every day at 13:00 UTC. The default
directory is `/opt/data`.

```
whakaaribn daemon
```

To use a custom directory:

```
whakaaribn daemon --directory /path/to/data
```
