from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from whakaaribn import (
    Bin,
    BinData,
    Discretizer,
    ForecastImputer,
    SequentialGroupSplit,
    convert_probability,
    get_color,
    hex_to_rgb,
    moving_average,
    pre_eruption_window,
)


def test_binning():
    dates = pd.date_range("13/11/2020", periods=9, freq="1D")
    df = pd.DataFrame({"RSAM": np.arange(9.0)}, index=dates)
    bd = BinData(df, "RSAM", 3, dropzeros=False)
    np.testing.assert_array_almost_equal(
        bd.marginals(), np.array([0.33, 0.33, 0.33]), 2
    )

    assert bd.binnames == bd._bin_names
    assert bd.bins == bd._bins

    # Test with nan's in the data
    df1 = df.copy()
    df1.iloc[0, 0] = np.nan
    bd1 = BinData(df1, "RSAM", 3, dropzeros=False)
    np.testing.assert_array_equal(
        bd1.marginals(), np.array([0.375, 0.25, 0.375]))

    np.testing.assert_array_equal(
        bd.query([1, 1, 5, 8.5, np.nan]),
        np.array(["state_0", "state_0", "state_1", "state_2", "*"]),
    )
    assert bd.query(1) == bd.binnames[0]
    with pytest.raises(ValueError):
        bd.query(-1, extrapolate=False)
    with pytest.raises(ValueError):
        bd.query(9, extrapolate=False)

    bd2 = BinData(df, "RSAM", [0, 3, 8], btype="size")
    assert bd2.query(3.1) == bd2.binnames[1]

    dates3 = pd.date_range("13/11/2020", periods=150, freq="1D")
    df3 = pd.DataFrame({"RSAM": np.arange(150.0)}, index=dates3)
    bd3 = BinData(df3, "RSAM", [0, 60, 90, 100], dropzeros=False)
    np.testing.assert_array_almost_equal(
        bd3.marginals(), np.array([0.6, 0.3, 0.1]))


def test_binning_uncertainty():
    """
    Vary bin boundaries by small amounts to estimate uncertainty.
    """
    dates = pd.date_range("13/11/2020", periods=9, freq="1D")
    df = pd.DataFrame({"RSAM": np.arange(9.0)}, index=dates)
    bd0 = BinData(df, "RSAM", 3, dropzeros=False, factor=0, seed=42)
    bd2 = BinData(df, "RSAM", 3, dropzeros=False, factor=0.5, seed=42)
    bd3 = BinData(
        df,
        "RSAM",
        3,
        names=["Low", "Medium", "High"],
        dropzeros=False,
        factor=2.0,
        seed=42,
    )
    np.testing.assert_array_almost_equal(
        bd0.marginals(), [0.333, 0.333, 0.333], 3)
    np.testing.assert_array_almost_equal(
        bd2.marginals(), [0.333, 0.555, 0.111], 3)
    np.testing.assert_array_almost_equal(
        bd3.marginals(), [0.303, 0.606, 0.091], 3)
    assert bd0.query(6.5) == bd0.binnames[2]
    assert bd2.query(6.5) == bd2.binnames[1]

    b0 = Bin([0, 1.74, 5.4, 1e10], ["Low", "Medium", "High"])
    assert b0.query(1.7) == "Low"
    b1 = Bin([0, 1.74, 5.4, 1e10], [
             "Low", "Medium", "High"], factor=0.5, seed=42)
    assert b1.query(1.7) == "Medium"


def test_binning_wo_data():
    b = Bin([0, 1.74, 5.4, 1e10], ["Low", "Medium", "High"])
    assert b.query(1.5) == "Low"
    b1 = Bin([1e10, 0.09, -0.38, -1e10],
             ["Increasing", "Unchanged", "Decreasing"])
    assert b1.query(1.0) == "Increasing"
    assert b1.query(-1.0) == "Decreasing"


def test_discretizer():
    """
    Test the scikit-learn compatible discretizer class.
    """
    dates = pd.date_range("13/11/2020", periods=9, freq="1D")
    df = pd.DataFrame({"RSAM": np.arange(9.0)}, index=dates)
    desc = Discretizer(bins=(0, 5, 95, 100))
    rv = desc.fit_transform(np.tile(np.arange(5)[:, np.newaxis], (1, 3)))
    np.testing.assert_equal(
        rv.iloc[:, 1].values, np.array([0, 1, 1, 1, 2])
    )
    desc1 = Discretizer(bins=(0, 5, 95, 100)).set_output(transform="pandas")
    rv1 = desc1.fit_transform(df)
    np.testing.assert_equal(
        rv1.iloc[:, 0].values, np.array(
            [0, 1, 1, 1, 1, 1, 1, 1, 2], dtype=float)
    )

    desc = Discretizer(bins=[0, 20, 40, 60, 80, 100])
    rv = desc.fit_transform(np.tile(np.arange(5)[:, np.newaxis], (1, 3)))
    np.testing.assert_equal(
        rv.iloc[:, 0].values, np.array([0, 1, 2, 3, 4])
    )


def test_moving_average():
    a = np.arange(18)
    ret = moving_average(a.reshape(6, 3), window_size=3, axis=0)
    assert ret.shape == (6, 3)
    ret = moving_average(a.reshape(6, 3), window_size=3, axis=1)
    assert ret.shape == (6, 3)
    ret = moving_average(a.reshape(6, 3), window_size=3, axis=1, nan=False)
    assert ret.shape == (6, 3)
    assert np.all(~np.isnan(ret))
    ret = moving_average(np.arange(18), window_size=3)
    assert ret.shape == (18,)


def test_get_color():
    matplotlib.style.use("bmh")
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    expected_color = f"rgba{hex_to_rgb(colors[0], alpha=1.0)}"

    assert get_color(0) == expected_color


def test_forecast_imputer():
    imp = ForecastImputer(sigmas=[1.0, 1.0])
    data = np.tile(np.r_[np.arange(10), np.nan, 20], (2, 1)).T
    x = imp.fit_transform(data)
    assert (x[-3, 0] < x[-2, 0]) & (x[-2, 0] < x[-1, 0])


def test_convert_probability():
    # Test case 1
    prob = np.array([0.1, 0.5, 0.8])
    hin = 28
    hnew = 91
    expected = np.array([0.29, 0.895, 0.995])
    result = convert_probability(prob, hin, hnew)
    np.testing.assert_array_almost_equal(result, expected, decimal=3)
    result1 = convert_probability(result, hnew, hin)
    np.testing.assert_array_almost_equal(result1, prob, decimal=3)


def test_sequential_group_split():
    groups = np.array(["a", "a", "a", "b", "b", "b",
                      "c", "c", "c", "d", "d", "d"])
    data = pd.DataFrame(
        {"x": np.arange(groups.size)},
        index=pd.date_range("2000-01-01", periods=groups.size),
    )
    sgs = SequentialGroupSplit(groups)
    i = 1
    for train, test in sgs.split(data):
        assert train.size == i * 3
        assert test.size == 3
        i += 1
    assert i == 4
    assert train.size == groups.size - 3
    assert sgs.get_n_splits() == 3


def test_pre_eruption_window():
    y = np.r_[np.zeros(9), 1]
    assert 5 == np.sum(pre_eruption_window(y, 5))
    # Make sure applying the transformation twice does not change the result
    assert 5 == np.sum(pre_eruption_window(y, 5))
