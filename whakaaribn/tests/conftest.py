import pytest

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
