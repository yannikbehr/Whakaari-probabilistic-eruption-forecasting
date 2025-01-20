from datetime import datetime
import hashlib
import unittest

import numpy as np
import pandas as pd

from whakaaribn import (eqRate,
                       load_ruapehu_earthquakes,
                       load_all_whakaari_data,
                       load_whakaari_lp,
                       load_whakaari_vlp)


class EQTestCase(unittest.TestCase):

    def test_load_catalogue(self):
        """
        Test loading the earthquake catalogue for
        different zones.
        """
        cat1, cat2 = load_ruapehu_earthquakes(enddate=datetime(2021,6,1),
                                              ignore_cache=False)
        self.assertEqual(cat1.shape[0], 11491)
        self.assertEqual(cat2.shape[0], 10289)

    def test_eqRate(self):
        """
        Test earthquake rate with a synthetic example.
        """
        cat = pd.DataFrame({'origintime': pd.date_range('1978-07-18',
                                                        periods=5,
                                                        freq='2D')})
        eqr1 = eqRate(cat, fixed_time=2, enddate=datetime(1978,7,27))
        eqr2 = eqRate(cat, fixed_nevents=2, enddate=datetime(1978,7,27))
        np.testing.assert_array_equal(eqr1.values.squeeze(),
                                      np.array([0.5]*6))
        np.testing.assert_array_equal(eqr2.values.squeeze(),
                                      np.array([0.5]*3))

    def test_whakaari_volcano_seismicity(self):
        """
        Test load LPs and VLPs
        """
        try:
            dflp = load_whakaari_lp(enddate=datetime(2021,6,1))
            dfvlp = load_whakaari_vlp(enddate=datetime(2021,6,1))
            hash_test_lp = 'ed70c87175a0da49aa3a019402135b27a7542910aa8b9cebf671403abcfe03fe'
            hash_test_vlp = '8ee67ffe869061a807b0c76f432b29b4a3d957df528177efdc1ea710a10b79dc'
            hash_lp = hashlib.sha256(pd.util.hash_pandas_object(dflp).values.tobytes()).hexdigest()
            hash_vlp = hashlib.sha256(pd.util.hash_pandas_object(dfvlp).values.tobytes()).hexdigest()
            self.assertEqual(hash_test_lp, hash_lp)
            self.assertEqual(hash_test_vlp, hash_vlp)
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main()

