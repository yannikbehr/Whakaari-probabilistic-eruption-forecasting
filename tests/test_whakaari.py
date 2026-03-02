import inspect
import os

import numpy as np
import pandas as pd
import pytest

from whakaaribn import (
    SequentialGroupSplit,
)
from whakaaribn.forecast import forecast, sensitivity_analysis, uncertainty_analysis


def test_forecasts(tmp_path, setup_simulated_data):
    data = setup_simulated_data
    xds = forecast(data, pew=30, bins=(0, 5, 20, 80, 95, 100), smoothing=30)
    assert xds.probs.shape[0] == 1000
    assert xds.probs.min() >= 0
    assert xds.probs.max() <= 1


def test_sensitivity_analysis(setup_real_data):
    data = setup_real_data

    xds_all = sensitivity_analysis(
        data, pew=30, bins=(0, 5, 20, 80, 95, 100), factor=0.1, nmodels=3
    )
    assert xds_all.shape[0] == 3


@pytest.mark.slow
def test_uncertainty_analysis(setup_data_dir, setup_real_data):

    data_dir = setup_data_dir
    search_results_file = os.path.join(data_dir, "grid_search_results.csv")
    search_results = pd.read_csv(
        search_results_file, index_col=(0, 1, 2), converters={"params": eval}
    )
    data = setup_real_data
    xds_uncertainty = uncertainty_analysis(data, search_results)
    assert xds_uncertainty.shape[0] == 3
