from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from whakaaribn import BinData, load_ruapehu_gas, load_whakaari_gas, load_whakaari_so2


@pytest.mark.webservice
def test_ruapehu():
    co2data = load_ruapehu_gas("co2", enddate=datetime(2020, 8, 31), ignore_cache=True)
    so2data = load_ruapehu_gas("so2", enddate=datetime(2020, 8, 31), ignore_cache=True)
    h2sdata = load_ruapehu_gas("h2s", enddate=datetime(2020, 8, 31), ignore_cache=True)

    co2b = BinData(co2data, "obs", bins=[0, 68.27, 95.45, 100], btype="freq")
    so2b = BinData(so2data, "obs", bins=[0, 68.27, 95.45, 100], btype="freq")
    h2sb = BinData(h2sdata, "obs", bins=[0, 68.27, 95.45, 100], btype="freq")

    np.testing.assert_array_almost_equal(co2b.marginals(), (0.682, 0.273, 0.045), 3)
    np.testing.assert_array_almost_equal(so2b.marginals(), (0.682, 0.271, 0.047), 3)
    np.testing.assert_array_almost_equal(h2sb.marginals(), (0.676, 0.268, 0.056), 3)

    assert co2b.query(500) == co2b.binnames[0]
    assert co2b.query(1000) == co2b.binnames[1]
    assert co2b.query(2100) == co2b.binnames[2]


@pytest.mark.webservice
def test_whakaari():
    co2data = load_whakaari_gas(
        "co2", method="contouring", enddate=datetime(2020, 8, 31), ignore_cache=True
    )
    so2data = load_whakaari_gas("so2", enddate=datetime(2020, 8, 31), ignore_cache=True)
    h2sdata = load_whakaari_gas(
        "h2s", method="contouring", enddate=datetime(2020, 8, 31), ignore_cache=True
    )

    co2b = BinData(co2data, "obs", bins=[0, 68.27, 95.45, 100], btype="freq")
    so2b = BinData(so2data, "obs", bins=[0, 68.27, 95.45, 100], btype="freq")
    h2sb = BinData(h2sdata, "obs", bins=[0, 68.27, 95.45, 100], btype="freq")

    np.testing.assert_array_almost_equal(co2b.marginals(), (0.684, 0.27, 0.046), 3)
    np.testing.assert_array_almost_equal(so2b.marginals(), (0.68, 0.275, 0.046), 3)
    np.testing.assert_array_almost_equal(h2sb.marginals(), (0.677, 0.274, 0.048), 3)

    assert co2b.query(1000) == co2b.binnames[0]
    assert co2b.query(2100) == co2b.binnames[1]
    assert co2b.query(3000) == co2b.binnames[2]


def generate_mock_data(enddate, startdate, ignore_cache):
    # Generate test data
    class SynData(object):
        def __init__(self, val=0, grad=0, noise_std=1.0, seed=None):
            self.grad = grad
            self.noise_std = noise_std
            self.val = val
            self.rs = np.random.default_rng(seed)

        def read(self, dt):
            self.val += self.grad * dt
            return self.val + self.rs.normal(0, self.noise_std)

    npts = 20
    R_std1 = 0.5
    R_std2 = 5.0
    dates = pd.date_range("1978-7-18", freq="1D", periods=npts)
    sensor1 = SynData(0, 2, noise_std=R_std1, seed=42)
    sensor2 = SynData(0, 2.3, noise_std=R_std2, seed=42)
    t1 = [sensor1.read(5) if dt % 5 == 0 else np.nan for dt in range(npts)]
    t1_err = [R_std1 if dt % 5 == 0 else np.nan for dt in range(npts)]
    t2 = [sensor2.read(1) for dt in range(npts)]
    t2_err = [R_std2 for dt in range(npts)]
    obs = pd.DataFrame({"WI100-cosp": t1, "WI301-mdoas-ch": t2}, index=dates)
    obs_err = pd.DataFrame(
        {"WI100-cosp": t1_err, "WI301-mdoas-ch": t2_err}, index=dates
    )
    return (obs, obs_err)


def test_whakaari_so2(monkeypatch):
    """
    Test loading SO2 data from Whakaari.
    """
    monkeypatch.setattr("whakaaribn.data._load_whakaari_so2", generate_mock_data)
    so2 = load_whakaari_so2()
    assert abs(so2["obs"].mean() - 27.32) < 0.1


@pytest.mark.webservice
def test_whakaari_so2_live():
    """
    Test loading SO2 data from Whakaari.
    """
    so2 = load_whakaari_so2(ignore_cache=True, fuse=True, enddate=datetime(2024, 1, 31))
    assert abs(so2["obs"].mean() - 237.7) < 0.1
