from io import StringIO

import numpy as np
import pandas as pd

from whakaaribn import convert_probability


def test_forecast(setup_api):
    client, l = setup_api
    params = dict(name='best_model',
                  group='whakaari_forecasts',
                  starttime=str(l.starttime),
                  endtime=str(l.endtime),
                  horizon=28)
    with client.stream("GET", "/forecast", params=params) as r:
        r.read()
        txt = r.text
    df = pd.read_csv(StringIO(txt), parse_dates=True, index_col=0)
    test_data = convert_probability(l('best_model').to_pandas().interpolate().values,
                                    40, 28)
    np.testing.assert_array_almost_equal(df['feature'].values,
                                         test_data)

def test_labels(setup_api):
    client, l = setup_api
    with client.stream("GET", "/labels") as r:
        r.read()
        txt = r.text
    retval = eval(txt)
    assert len(retval) == 19 
    assert retval[0]['time'] == 1344038400000 

    params = dict(starttime="2019-12-01T00:00:00Z",
                  endtime="2025-01-01T00:00:00Z")
    with client.stream("GET", "/labels", params=params) as r:
        r.read()
        txt = r.text
    retval = eval(txt)
    assert len(retval) == 8 
    assert retval[0]['time'] == 1575849600000 
 