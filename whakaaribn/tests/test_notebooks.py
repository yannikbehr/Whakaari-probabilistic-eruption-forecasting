import inspect
import os
import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class NBTestCase(unittest.TestCase):

    def setUp(self):
        self.cwd = os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe())))
        self.nbdir = os.path.join(self.cwd, '..', 'examples')

    def test_ruapehu_manuscript(self):
        try:
            nbfn = os.path.join(self.nbdir, 'Ruapehu_manuscript.ipynb')
            with open(nbfn) as f:
                nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=1200, kernel_name='whakaaribn')
            ep.preprocess(nb, {'metadata': {'path': self.nbdir}}) 
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main()
