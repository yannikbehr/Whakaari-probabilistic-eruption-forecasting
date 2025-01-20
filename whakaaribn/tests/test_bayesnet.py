import inspect
import os
import unittest
from datetime import datetime as dt

import numpy as np
import pandas as pd

from whakaaribn import (
    BinData,
    Prior,
    gradient,
    load_all_ruapehu_data,
    load_rcl_temperature,
    load_ruapehu_catalogue,
    load_ruapehu_earthquakes,
    load_ruapehu_water_chemistry,
)
from whakaaribn.ruapehu import RuapehuModel, data_table, old_data_table


class BNTestCase(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
            "data",
        )

    def test_set_bins(self):
        """
        Test updating bins.
        """
        bins = {"RSAM": {"bins": [0, 68.27, 95.45, 100], "btype": "freq"}}

        rnet = RuapehuModel(91, enddate=dt(2020, 8, 24), version=2)
        old_props = rnet.props.copy()
        rnet.set_bins(bins)
        self.assertEqual(rnet.props["RSAM"]["bins"], bins["RSAM"]["bins"])
        self.assertEqual(rnet.props["RSAM"]["states"], old_props["RSAM"]["states"])

    def test_load_template_error(self):
        """
        Test the correct error is raised if the requested
        template file does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            RuapehuModel(91, enddate=dt(2020, 8, 24), version=123)

    def test_bn_version1(self):
        """
        Test version 1 of the Ruapehu Bayesian Network.
        """
        bins = {
            "TemperatureBin": {"bins": (9.0, 24.0, 31.0, 60.0), "btype": "size"},
            "GradientBin": {"bins": (-25.5, 0.0, 0.1, 23.5), "btype": "size"},
            "RSAM": {"bins": [0, 68.27, 95.45, 100], "btype": "freq"},
            "SO2": {"bins": [0, 68.27, 95.45, 100], "btype": "freq"},
            "CO2": {"bins": [0, 68.27, 95.45, 100], "btype": "freq"},
            "H2S": {"bins": [0, 68.27, 95.45, 100], "btype": "freq"},
        }

        rm = RuapehuModel(91, enddate=dt(2020, 8, 24), version=1)
        rm.set_bins(bins)
        rnet = rm()
        rnet.net.update_beliefs()
        posteriors1 = rnet.net.get_node_value("Eruption")
        yes = 0.08
        no = 0.92
        self.assertAlmostEqual(posteriors1[0], yes, 3)
        self.assertAlmostEqual(posteriors1[1], no, 3)
        rnet.set_evidence("TemperatureBin", 40.0)
        posteriors2 = rnet.net.get_node_value("Eruption")
        yes = 0.126
        no = 0.874
        self.assertAlmostEqual(posteriors2[0], yes, 3)
        self.assertAlmostEqual(posteriors2[1], no, 3)
        # long-term probability from earthquake rate
        prior = Prior("ruapehu", 3, "91 days", enddate=dt(2020, 8, 24))
        self.assertAlmostEqual(prior.sample(), 0.067, 3)

    def test_bn_version2(self):
        """
        Test version 2 of the Ruapehu Bayesian Network.
        """
        bins = {
            "TemperatureBin": {"bins": (9.0, 24.0, 31.0, 60.0), "btype": "size"},
            "GradientBin": {"bins": (-25.5, 0.0, 0.1, 23.5), "btype": "size"},
            "RSAM": {"bins": [0, 68.27, 95.45, 100], "btype": "freq"},
            "SO2": {"bins": [0, 68.27, 95.45, 100], "btype": "freq"},
            "CO2": {"bins": [0, 68.27, 95.45, 100], "btype": "freq"},
            "H2S": {"bins": [0, 68.27, 95.45, 100], "btype": "freq"},
            "Mg_ClBin": {"bins": 3, "btype": "freq"},
            "SO4": {"bins": 3, "btype": "freq"},
        }

        rm = RuapehuModel(91, enddate=dt(2020, 8, 24), version=2)
        rm.set_bins(bins)
        rnet = rm()
        rnet.net.update_beliefs()
        posteriors1 = rnet.net.get_node_value("Eruption")
        yes = 0.08
        no = 0.92
        self.assertAlmostEqual(posteriors1[0], yes, 2)
        self.assertAlmostEqual(posteriors1[1], no, 2)
        rnet.set_evidence("TemperatureBin", 40.0)
        posteriors2 = rnet.net.get_node_value("Eruption")
        yes = 0.122
        no = 0.878
        self.assertAlmostEqual(posteriors2[0], yes, 3)
        self.assertAlmostEqual(posteriors2[1], no, 3)
        # long-term probability from earthquake rate
        prior = Prior("ruapehu", 3, "91 days", enddate=dt(2020, 8, 24))
        self.assertAlmostEqual(prior.sample(), 0.067, 3)

    def test_data_table(self):
        T = load_rcl_temperature(
            interpolate="linear", startdate="2007-1-1", enddate="2008-1-1"
        )
        Tb = BinData(
            T,
            "obs",
            bins=[T["obs"].min(), 24, 31, T["obs"].max()],
            btype="size",
            dropzeros=False,
        )
        dT = gradient(T)
        dTb = BinData(
            dT,
            "obs",
            bins=[dT["obs"].min(), 0, 0.1, dT["obs"].max()],
            btype="size",
            dropzeros=False,
        )
        cl = load_ruapehu_water_chemistry("Cl", interpolate="linear")
        mg = load_ruapehu_water_chemistry("Mg", interpolate="linear")
        mgcl = mg / cl
        dmgcl = gradient(mgcl, period="90D")
        dmgcl = dmgcl[dmgcl.index > "1967-01-01"]
        dmgcl.obs = np.where(dmgcl.obs > 0.0015, 0.0015, dmgcl.obs)
        dmgclb = BinData(
            dmgcl, "obs", 4, btype="freq", limits=(-0.0015, 0.0015), dropzeros=False
        )
        cat1, cat2 = load_ruapehu_earthquakes()
        # eqr1_df = eqRate(cat1, fixed_time=20)
        # eqr2_df = eqRate(cat2, fixed_time=20)
        # eqr_b1 = BinData(eqr1_df, 'obs', nbins, btype='freq', limits=(0,10))
        # eqr_b2 = BinData(eqr2_df, 'obs', nbins, btype='freq', limits=(0,10))
        params = [("Temperature", Tb), ("Gradient", dTb)]
        eruptions = load_ruapehu_catalogue(3, "14D")
        data = data_table(params, eruptions, interval=91)
        test_data = old_data_table(Tb, dTb, eruptions)
        test_data.index.freq = "D"
        pd.testing.assert_series_equal(data["Temperature"], test_data["Temperature"])
        pd.testing.assert_series_equal(data["Gradient"], test_data["Gradient"])
        pd.testing.assert_series_equal(
            data["MaxTemperature"], test_data["MaxTemperature"]
        )
        pd.testing.assert_series_equal(
            data["MaxTemperatureBin"], test_data["MaxTemperatureBin"]
        )
        pd.testing.assert_series_equal(
            data["MinTemperature"], test_data["MinTemperature"]
        )
        pd.testing.assert_series_equal(
            data["TemperatureBin"], test_data["TemperatureBin"]
        )
        pd.testing.assert_series_equal(data["Eruption"], test_data["Eruption"])

    def test_data_table_slow(self):
        """
        Test data table with longer date range
        """
        T = load_rcl_temperature(interpolate="linear", enddate="2020-8-24")
        Tb = BinData(
            T,
            "obs",
            bins=[T["obs"].min(), 24, 31, T["obs"].max()],
            btype="size",
            dropzeros=False,
        )
        dT = gradient(T)
        dTb = BinData(
            dT,
            "obs",
            bins=[dT["obs"].min(), 0, 0.1, dT["obs"].max()],
            btype="size",
            dropzeros=False,
        )
        params = [("Temperature", Tb), ("Gradient", dTb)]
        eruptions = load_ruapehu_catalogue(3, "14D")
        data = data_table(params, eruptions, interval=91)
        # remove NaNs to make it look like the original data
        ndata = data[~np.isnan(data.Temperature)]
        test_data = old_data_table(Tb, dTb, eruptions)
        pd.testing.assert_series_equal(ndata["Temperature"], test_data["Temperature"])
        pd.testing.assert_series_equal(ndata["Gradient"], test_data["Gradient"])
        pd.testing.assert_series_equal(
            ndata["MaxTemperature"], test_data["MaxTemperature"]
        )
        pd.testing.assert_series_equal(
            ndata["MaxTemperatureBin"], test_data["MaxTemperatureBin"]
        )
        pd.testing.assert_series_equal(
            ndata["MinTemperature"], test_data["MinTemperature"]
        )
        pd.testing.assert_series_equal(
            ndata["TemperatureBin"], test_data["TemperatureBin"]
        )
        pd.testing.assert_series_equal(ndata["Eruption"], test_data["Eruption"])

    def test_eqr_bug(self):
        enddate = dt(2014, 1, 28)
        rm = RuapehuModel(91, enddate=enddate, version=2)
        with self.assertWarns(RuntimeWarning):
            rnet = rm()

    def test_all_data(self):
        """
        Test generating a dataframe containing all observational data.
        """
        enddate = dt(2021, 11, 3, 0, 0, 0)
        df = load_all_ruapehu_data(enddate=enddate)
        self.assertEqual(df["CO2"].isna().sum(), 19445)
        df = load_all_ruapehu_data(fill_method=None, enddate=enddate)
        self.assertEqual(df["CO2"].isna().sum(), 26057)
        df = load_all_ruapehu_data(fill_method="ffill", enddate=enddate)
        self.assertEqual(df["CO2"].isna().sum(), 26045)
        with self.assertRaises(ValueError):
            df = load_all_ruapehu_data(fill_method="blub")


if __name__ == "__main__":
    unittest.main()
