import os

import numpy as np
import pandas as pd
import pytest
from aitana import whakaari
from sklearn.pipeline import Pipeline

from whakaaribn import Discretizer, split_by_group
from whakaaribn.grid_search import (
    evaluate_threshold,
    get_best_estimator,
    grid_search,
    make_strictly_increasing,
    my_roc_auc,
)
from whakaaribn.model import WhakaariModel


def test_make_strictly_increasing():
    seq = [0.1, 0.2, 0.3, 0.4, 0.2, 0.6, 0.7, 0.5, 1]
    assert make_strictly_increasing(seq) == [0.1, 0.2, 0.3, 0.4, 0.4, 0.6, 0.7, 0.7, 1]


def test_evaluate_threshold():
    test_seq = np.r_[np.ones(30) * 0.6, np.ones(30) * 0.2, np.ones(40) * 0.7]
    dates = pd.date_range(
        pd.Timestamp("2013-10-21") - pd.Timedelta(days=100), periods=100, tz="UTC"
    )
    test_df = pd.DataFrame({"prob": test_seq}, index=dates)
    eruption_df = pd.DataFrame(
        {"Activity_Scale": [2]}, index=[pd.Timestamp("2013-10-11", tz="UTC")]
    )
    test_df["eruptions"] = eruption_df.reindex(test_df.index, fill_value=0)[
        "Activity_Scale"
    ]
    result, windows = evaluate_threshold(
        0.5, test_df, debug=False, pew=None, return_windows=True
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
    test_df = pd.DataFrame({"prob": test_seq}, index=dates)
    test_df["eruptions"] = eruption_df.reindex(test_df.index, fill_value=0)[
        "Activity_Scale"
    ]
    result, windows = evaluate_threshold(
        0.5, test_df, debug=False, pew=None, return_windows=True
    )
    assert result["tp"] == 0
    assert result["fp"] == 30
    assert result["tn"] == 0
    assert result["fn"] == 70


def test_eruption_info(setup_real_data):
    """
    Ensure that the same eruption info can be retrieved by the following two methods.
    """
    data = setup_real_data
    eruptions_method1 = whakaari.eruptions(2, "0D", end_date=data.index[-1])

    eruptions_method2 = data["eruptions"].loc[data["eruptions"] > 0]
    assert np.all(eruptions_method1.sum() == eruptions_method2.sum())


def test_roc_auc(setup_real_data):
    data = setup_real_data
    pipe = Pipeline(
        [
            ("discretize", Discretizer()),
            ("clf", WhakaariModel(smoothing=30, uniformize=True, pew=0)),
        ]
    )
    pipe.set_output(transform="pandas")
    X_train, y_train, X_remainder, y_remainder = split_by_group(data, group="e")
    pipe.fit(X_train.ffill(), y_train)
    eruptions = whakaari.eruptions(2, "0D", end_date=data.index[-1])
    score = my_roc_auc(pipe, X_train.ffill(), None, eruptions=eruptions, use_pew=False)
    assert abs(score - 0.59) < 0.005


# @pytest.mark.slow
def test_grid_search(setup_simulated_data):
    data = setup_simulated_data
    params_gcv = [
        {
            "discretize__bins": [(0, 25, 50, 75, 100), (0, 20, 50, 80, 100)],
            "clf__nstates": [4],
            "clf__pew": np.arange(10, 30, 10),
        },
        {
            "discretize__bins": [(0, 5, 20, 80, 95, 100), (0, 5, 25, 75, 95, 100)],
            "clf__nstates": [5],
            "clf__pew": np.arange(10, 30, 10),
        },
    ]
    search_result = grid_search(data, params_gcv, njobs=10)
    assert len(search_result) == 8


def test_best_estimator(setup_data_dir):
    data_dir = setup_data_dir
    search_results = os.path.join(data_dir, "grid_search_results.csv")
    bins, pew = get_best_estimator(search_results)
    assert bins == [0, 5, 20, 80, 95, 100]
    assert pew == 80
