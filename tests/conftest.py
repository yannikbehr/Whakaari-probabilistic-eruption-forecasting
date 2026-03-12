import inspect
import os
from datetime import datetime, timezone

import pytest
from aitana import whakaari
from fastapi.testclient import TestClient
from tonik import Storage, generate_test_data

from whakaaribn import assign_group_labels
from whakaaribn.model import WhakaariModel


def pytest_addoption(parser):
    parser.addoption(
        "--runwebservice",
        action="store_true",
        default=False,
        help="run webservice tests",
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "webservice: mark tests that test webserivces")
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runwebservice"):
        skip_webservice = pytest.mark.skip(
            reason="need --runwebservice option to run")
        for item in items:
            if "webservice" in item.keywords:
                item.add_marker(skip_webservice)
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture()
def setup(tmp_path_factory):
    features1D = [
        "best_model",
        "median_sensitivity",
        "max_sensitivity",
        "min_sensitivity",
        "median_ensemble",
        "max_ensemble",
        "min_ensemble",
    ]

    savedir = tmp_path_factory.mktemp("whakaari_test_tmp", numbered=True)
    g = Storage("whakaari_forecasts", rootdir=savedir)
    tstart = datetime(2023, 1, 1)
    ndays = 10
    # Generate some fake data
    for _f in features1D:
        data = generate_test_data(
            tstart=tstart, feature_names=[_f], ndays=ndays, add_nans=False
        )
        data = (data - data.min()) / (data.max() - data.min())
        feat = data
        g.save(feat)
    return savedir, g


@pytest.fixture()
def setup_api(setup):
    savedir, g = setup
    from whakaaribn.api import Forecast

    ta = Forecast(str(savedir))
    client = TestClient(ta.app)
    g.starttime = datetime(2023, 1, 1)
    g.endtime = datetime(2023, 1, 6)
    return client, g


@pytest.fixture()
def setup_simulated_data():
    ml = WhakaariModel(randomize=True, seed=42)
    data = ml.simulate(n_samples=1000, mode="continuous")
    # assign groups 'a' to 'e' to the data, each group has 200 samples
    data["group"] = ["a"] * 200 + ["b"] * 200 + \
        ["c"] * 200 + ["d"] * 200 + ["e"] * 200
    return data


@pytest.fixture()
def setup_real_data():
    start_date = datetime(2009, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    data = whakaari.load_all(
        fill_method=None,
        start_date=start_date,
        end_date=end_date,
        ignore_data=("LP", "VLP"),
        fuse_so2=False,
    )
    eruptions = whakaari.eruptions(2, "0D", end_date=end_date)
    data_with_groups = assign_group_labels(
        data,
        eruptions,
        startdate=start_date,
        enddate=end_date,
        ndays=30,
        min_interval=360,
    )
    return data_with_groups


@pytest.fixture()
def setup_data_dir(request):
    data_dir = os.path.join(str(request.config.rootdir), "tests", "data")
    return data_dir
