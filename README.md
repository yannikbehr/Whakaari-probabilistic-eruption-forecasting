 [![pipeline status](https://git.gns.cri.nz/behrya/whakaaribn/badges/master/pipeline.svg)](https://git.gns.cri.nz/behrya/whakaaribn/-/commits/master)


## Probabilistic decision making for volcano monitoring
This is a Python package to forecast the probability of
an eruption given expert opinion and volcano monitoring data
using a Bayesian Network.

It requires input data from the [FITS](https://fits.geonet.org.nz/) webservice.
Other data, such as RSAM, comes with the package. The resulting Bayesian
Network can be accessed at http://volcanolab.gns.cri.nz:8081.

Maintainer: [Yannik Behr](mailto:y.behr@gns.cri.nz)

## Install

### Setup the conda environment
```
conda env create -f environment.yml
```

### Install the package
```
conda activate whakaaribn
python setup.py develop
```

## Operations
### External dependencies
The following has to run on volcanolab.gns.cri.nz
  * docker volume: `bayesbox_data`
  * docker image: `bayesbox`
 
## Contributions

* Clone the repository

```
git clone repo_name
```

* Create a new branch

```
git checkout -b branch_name
```

* Fix a bug or add a feature
* Commit changes and push the branch to the repository:

```
git add changed_or_new_file.py
git commit -m "some message"
git push -U origin branch_name
```

Confirm the new output is correct


Request code review


Merge branch into master/main branch
