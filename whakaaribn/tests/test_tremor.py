from datetime import datetime
import unittest

import numpy as np
import pandas as pd

from whakaaribn import BinData, load_whakaari_rsam


class TremorTestCase(unittest.TestCase):

    def test_whakaari(self):
        td = load_whakaari_rsam(enddate=datetime(2022,5,31))
        self.assertEqual(td.index[-1], pd.Timestamp(2022, 5, 31, 00, 00, 00))
        self.assertEqual(td.index[0], pd.Timestamp(2007, 4, 20))
        tdb = BinData(td, 'obs', bins=[0, 750, 1500, 5000],
                      btype='size', limits=(0,5000))
        np.testing.assert_array_almost_equal(tdb.marginals(),
                                             (0.94, 0.048, 0.013), 3)
        self.assertEqual(tdb.query(100), tdb.binnames[0]) 
        self.assertEqual(tdb.query(800), tdb.binnames[1]) 
        self.assertEqual(tdb.query(2500), tdb.binnames[2]) 

if __name__ == '__main__':
    unittest.main()
