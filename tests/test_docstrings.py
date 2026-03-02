import unittest
import doctest
import whakaaribn.util


def load_tests(tests):
    tests.addTests(doctest.DocTestSuite(whakaaribn.util))
    return tests


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    suite = unittest.TestSuite()
    runner.run(load_tests(suite))