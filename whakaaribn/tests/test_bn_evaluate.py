from datetime import datetime

import numpy as np
import pytest

from whakaaribn import load_all_ruapehu_data
from whakaaribn.ruapehu import bn_evaluate


@pytest.mark.slow
def test_uncertainty():
    """
    Test estimating confidence intervals
    """
    data = load_all_ruapehu_data(
        fill_method="interpolate", enddate=datetime(2021, 12, 31)
    )
    date0 = datetime(2021, 12, 1)
    rs = bn_evaluate(data, date0, progressbar=False, nmodels=10, seed=42)
    rmse = np.sqrt(np.sum((rs.emean - rs.econtrol) ** 2))
    assert abs(rmse - 0.024) < 0.01
