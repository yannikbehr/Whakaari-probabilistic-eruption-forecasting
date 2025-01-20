import unittest
from datetime import datetime as dt

from whakaaribn.prior import NodeBaseClass, Prior


class NodesTestCase(unittest.TestCase):
    def test_base_class(self):
        class NewClass(NodeBaseClass):
            def __init__(self, *args):
                super(NewClass, self).__init__(*args)

            def compute_probabilities(self):
                pass

        nc = NewClass("White Island", 1, "7 days", "blub")
        self.assertEqual(nc.volcano, "white island")
        self.assertEqual(nc.interval, "7 days")
        self.assertEqual(nc.eruption_scale, 1)
        self.assertEqual(nc.name, "blub")

        with self.assertRaises(TypeError):
            nc = NewClass(1, 2)

        with self.assertRaises(ValueError):
            nc = NewClass("Tongariro", 1, "7 days", "blub")

        with self.assertRaises(ValueError):
            nc = NewClass("White Island", 7, "7 days", "blub")

        with self.assertRaises(ValueError):
            nc = NewClass("White Island", 2, "12 days", "blub")

    def test_prior(self):
        prior = Prior("ruapehu", 3, "91 days", "91 days", enddate=dt(2020, 5, 1))
        self.assertAlmostEqual(prior.sample(), 0.054, 3)


if __name__ == "__main__":
    unittest.main()
