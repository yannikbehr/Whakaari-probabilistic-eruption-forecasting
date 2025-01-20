import inspect
import os

import numpy as np
import pandas as pd
import pytest

from whakaaribn import (
    SequentialGroupSplit,
    pre_eruption_window,
    WhakaariForecasts
)


@pytest.fixture(scope="session")
def setup():
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
        "data",
    )
    return data_dir


def test_pre_eruption_window():
    y = np.r_[np.zeros(9), 1]
    assert 5 == np.sum(pre_eruption_window(y, 5))
    # Make sure applying the transformation twice does not change the result
    assert 5 == np.sum(pre_eruption_window(y, 5))


def test_sequential_group_split():
    groups = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"])
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
    assert train.size == groups.size - 3
    assert sgs.get_n_splits() == 3

@pytest.mark.slow
def test_whakaari_model(tmp_path_factory):

    wf = WhakaariForecasts(args.outdir)
    wf.ensemble_forecasts(args.pew)


