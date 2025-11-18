import os

import numpy as np
import pandas as pd
from whakaaribn.grid_search import (
    make_strictly_increasing,
    evaluate_threshold,
    grid_search
    )


def test_make_strictly_increasing():
    seq = [.1, .2, .3, .4, .2, .6, .7, .5, 1]
    assert make_strictly_increasing(seq) == [.1, .2, .3, .4, .4, .6, .7, .7, 1]


def test_evaluate_threshold():
    test_seq = np.r_[np.ones(30)*.6, np.ones(30)*.2, np.ones(40)*.7]
    dates = pd.date_range(pd.Timestamp('2013-10-21') - pd.Timedelta(days=100), periods=100, tz='UTC')
    test_df = pd.Series(data=test_seq, index=dates)
    eruption_df = pd.DataFrame({'Activity_Scale': [2]}, index=[pd.Timestamp('2013-10-11', tz='UTC')])
    result, windows = evaluate_threshold(0.5, test_df, eruption_df, debug=False,
                                         pew=None, return_windows=True)
    assert result['tp'] == 40
    assert result['fp'] == 30 
    assert result['tn'] == 30
    assert result['fn'] == 0
    assert len(windows) == 3 
    for win in windows:
        if win['type'] == 'true_positive':
            assert win['end'] == pd.Timestamp('2013-10-20', tz='UTC')
            assert (win['end'] - win['start']) == pd.Timedelta(days=39)

    test_seq = np.r_[np.ones(30)*.7, np.ones(30)*.2, np.ones(40)*.4]
    dates = pd.date_range(pd.Timestamp('2013-10-21') - pd.Timedelta(days=100), periods=100, tz='UTC')
    test_df = pd.Series(data=test_seq, index=dates)
    result, windows = evaluate_threshold(0.5, test_df, eruption_df, debug=False,
                                         pew=None, return_windows=True)
    assert result['tp'] == 0
    assert result['fp'] == 30 
    assert result['tn'] == 0
    assert result['fn'] == 70


def test_grid_search(tmp_path_factory):
    # Create a temporary directory for the test
    tmp_dir = tmp_path_factory.mktemp("test_grid_search")
    params_gcv = {4: {  "discretize__bins": [(0, 25, 50, 75, 100), (0, 20, 50, 80, 100)],
                        "clf__hidden_nodes": [False],
                        "clf__uniformize": [True],
                        "clf__modelfile": [os.path.join(tmp_dir, 'fully_connected_model_4_states.xdsl')]
                    },
                  5: {  "discretize__bins": [(0, 5, 20, 80, 95, 100), (0, 5, 25, 75, 95, 100)],
                        "clf__hidden_nodes": [False],
                        "clf__uniformize": [True],
                        "clf__modelfile": [os.path.join(tmp_dir, 'fully_connected_model_5_states.xdsl')]
                    }
    }
    search_result = grid_search(params_gcv, njobs=1, outputdir=tmp_dir)
    print(search_result)
