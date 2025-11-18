from datetime import datetime

from fastapi.testclient import TestClient
import pytest
from tonik import Storage, generate_test_data 


def pytest_addoption(parser):
    parser.addoption(
        "--runwebservice", action="store_true", default=False, help="run webservice tests"
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "webservice: mark tests that test webserivces")
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runwebservice"):
        skip_webservice = pytest.mark.skip(reason="need --runwebservice option to run")
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
    features1D = ['best_model',
                  'median_sensitivity',
                  'max_sensitivity',
                  'min_sensitivity',
                  'median_ensemble',
                  'max_ensemble',
                  'min_ensemble']

    savedir = tmp_path_factory.mktemp('whakaari_test_tmp', numbered=True)
    g = Storage('whakaari_forecasts', rootdir=savedir)
    tstart = datetime(2023, 1, 1)
    ndays = 10
   # Generate some fake data
    for _f in features1D:
        data = generate_test_data(tstart=tstart,
                                    feature_names=[_f],
                                    ndays=ndays, add_nans=False)
        data = (data - data.min())/(data.max() - data.min())
        feat = data 
        g.save(feat)
    return savedir, g

@pytest.fixture()
def setup_api(setup):
    savedir, g = setup 
    from whakaaribn import Forecast
    ta = Forecast(str(savedir))
    client = TestClient(ta.app)
    g.starttime = datetime(2023, 1, 1)
    g.endtime = datetime(2023, 1, 6)
    return client, g