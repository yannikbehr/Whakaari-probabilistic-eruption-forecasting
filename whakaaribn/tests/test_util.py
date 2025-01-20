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
    convert_probability,
    eqRate,
    get_color,
    hash_dataframe,
    hex_to_rgb,
    load_ruapehu_earthquakes,
    moving_average,
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
    np.testing.assert_array_equal(bd1.marginals(), np.array([0.375, 0.25, 0.375]))

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
    np.testing.assert_array_almost_equal(bd3.marginals(), np.array([0.6, 0.3, 0.1]))


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
    np.testing.assert_array_almost_equal(bd0.marginals(), [0.333, 0.333, 0.333], 3)
    np.testing.assert_array_almost_equal(bd2.marginals(), [0.333, 0.555, 0.111], 3)
    np.testing.assert_array_almost_equal(bd3.marginals(), [0.303, 0.606, 0.091], 3)
    assert bd0.query(6.5) == bd0.binnames[2]
    assert bd2.query(6.5) == bd2.binnames[1]

    b0 = Bin([0, 1.74, 5.4, 1e10], ["Low", "Medium", "High"])
    assert b0.query(1.7) == "Low"
    b1 = Bin([0, 1.74, 5.4, 1e10], ["Low", "Medium", "High"], factor=0.5, seed=42)
    assert b1.query(1.7) == "Medium"


def test_binning_wo_data():
    b = Bin([0, 1.74, 5.4, 1e10], ["Low", "Medium", "High"])
    assert b.query(1.5) == "Low"
    b1 = Bin([1e10, 0.09, -0.38, -1e10], ["Increasing", "Unchanged", "Decreasing"])
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
        rv.iloc[:, 1].values, np.array(["low", "medium", "medium", "medium", "high"])
    )
    assert type(rv) == pd.core.frame.DataFrame
    desc1 = Discretizer(bins=(0, 5, 95, 100)).set_output(transform="pandas")
    rv1 = desc1.fit_transform(df)
    assert type(rv1) == pd.core.frame.DataFrame
    assert rv1.columns[0] == "RSAM"


def test_eq_rate():
    """
    Test computing earthquake rates from the Ruapehu catalogue.
    """
    cat_outer, cat_inner = load_ruapehu_earthquakes(
        startdate=datetime(2021, 12, 3),
        enddate=datetime(2022, 1, 3),
        ignore_cache=False,
    )
    eqr_inner = eqRate(cat_inner, fixed_time=7).resample("D").mean().interpolate()
    assert abs(eqr_inner.values.mean() - 1.67) < 0.01
    # Mix up the order of the catalogue to make sure this still works
    cat_inner.sort_values("magnitude", inplace=True)
    eqr_inner = eqRate(cat_inner, fixed_time=7).resample("D").mean().interpolate()
    assert abs(eqr_inner.values.mean() - 1.67) < 0.01


def test_moving_average():
    a = np.arange(18)
    ret = moving_average(a.reshape(6, 3), window_size=3, axis=0)
    assert ret.shape == (6, 3)
    ret = moving_average(a.reshape(6, 3), window_size=3, axis=1)
    assert ret.shape == (6, 3)
    ret = moving_average(a.reshape(6, 3), window_size=3, axis=1, nan=False)
    assert ret.shape == (6, 3)
    assert np.alltrue(~np.isnan(ret))
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


def test_hash_dataframe():
    df = pd.DataFrame(np.tile(np.r_[np.arange(10), np.nan, 20], (2, 1)).T)
    hash_ = hash_dataframe(df)
    assert hash_ == "0a726f2609a2b1033f378b5b30636ac65869594cea95eaa6c6e8c836f8e9bffb"


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
