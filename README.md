## Probabilistic eruption forecasting for Whakaari/White Island using Bayesian Networks
This is the source code accompanying the manuscript on probabilistic eruption forecasting for Whakaari/White Island using Bayesian Networks. A pre-print of the manuscript can be found [here](https://essopenarchive.org/users/820715/articles/1219096-probabilistic-multi-sensor-eruption-forecasting).

Maintainer: [Yannik Behr](mailto:y.behr@gns.cri.nz)

## Installation

### Dependencies
The python package depends on the SMILE library for Bayesian Network learning and inference. The library is written and maintained by [BayesFusion](https://www.bayesfusion.com/) and is free for academic users.

### Download the source code using git

```
git clone https://github.com/yannikbehr/whakaari_eruption_forecasting.git
cd whakaari_eruption_forecasting
```

### Setup the conda environment
```
conda env create -f environment.yml
conda activate whakaaribn
```
### Installing the PYSMILE library
```
pip install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile
```
Next obtain a license from [BayesFusion](https://www.bayesfusion.com/).
Academic users can obtain a free license, all other users can get a 30-day evaluation license
here: (https://download.bayesfusion.com/files.html?category=Business).
Once you have unzipped the package with license files run the following command to copy the license key to the correct place:

```
cp pysmile_license.py $(python -c "import site; print(site.getsitepackages()[0]))
```

### Install the package
```
pip install .
```

### Run the Jupyter notebook
First install a jupyter notebook server
```
pip install notebook
```

Then start the notebook server
```
jupyter notebook
```

And finally open the notebook `Whakaari_BN.ipynb` located under the `/examples` directory.
To generate all the figures from the manuscript go to `Run` -> `Run All Cells`.
