import os

import numpy as np
import pandas as pd
import pytest

from whakaaribn.grid_search import (
    evaluate_threshold,
    get_best_estimator,
    grid_search,
    make_strictly_increasing,
)


def test_make_strictly_increasing():
    seq = [0.1, 0.2, 0.3, 0.4, 0.2, 0.6, 0.7, 0.5, 1]
    assert make_strictly_increasing(
        seq) == [0.1, 0.2, 0.3, 0.4, 0.4, 0.6, 0.7, 0.7, 1]


def test_evaluate_threshold():
    test_seq = np.r_[np.ones(30) * 0.6, np.ones(30) * 0.2, np.ones(40) * 0.7]
    dates = pd.date_range(
        pd.Timestamp("2013-10-21") - pd.Timedelta(days=100), periods=100, tz="UTC"
    )
    test_df = pd.Series(data=test_seq, index=dates)
    eruption_df = pd.DataFrame(
        {"Activity_Scale": [2]}, index=[pd.Timestamp("2013-10-11", tz="UTC")]
    )
    result, windows = evaluate_threshold(
        0.5, test_df, eruption_df, debug=False, pew=None, return_windows=True
    )
    assert result["tp"] == 40
    assert result["fp"] == 30
    assert result["tn"] == 30
    assert result["fn"] == 0
    assert len(windows) == 3
    for win in windows:
        if win["type"] == "true_positive":
            assert win["end"] == pd.Timestamp("2013-10-20", tz="UTC")
            assert (win["end"] - win["start"]) == pd.Timedelta(days=39)

    test_seq = np.r_[np.ones(30) * 0.7, np.ones(30) * 0.2, np.ones(40) * 0.4]
    dates = pd.date_range(
        pd.Timestamp("2013-10-21") - pd.Timedelta(days=100), periods=100, tz="UTC"
    )
    test_df = pd.Series(data=test_seq, index=dates)
    result, windows = evaluate_threshold(
        0.5, test_df, eruption_df, debug=False, pew=None, return_windows=True
    )
    assert result["tp"] == 0
    assert result["fp"] == 30
    assert result["tn"] == 0
    assert result["fn"] == 70


@pytest.mark.slow
def test_grid_search(setup_simulated_data):
    data = setup_simulated_data
    params_gcv = {
        4: {
            "discretize__bins": [(0, 25, 50, 75, 100), (0, 20, 50, 80, 100)],
            "clf__nstates": [4, 4],
        },
        5: {
            "discretize__bins": [(0, 5, 20, 80, 95, 100), (0, 5, 25, 75, 95, 100)],
            "clf__nstates": [5, 5],
        },
    }
    search_result = grid_search(data, params_gcv, njobs=1, pews=[10, 20])
    assert len(search_result) == 16


def test_best_estimator(setup_data_dir):
    data_dir = setup_data_dir
    search_results = os.path.join(data_dir, "grid_search_results.csv")
    bins, pew = get_best_estimator(search_results)
    assert bins == [0, 5, 20, 80, 95, 100]
    assert pew == 80
