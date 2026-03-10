import math
import os
import tempfile
from collections import OrderedDict
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator

try:
    import pysmile
    import pysmile_license
    from pysmile import SMILEException
    PYSMILE_AVAILABLE = True
except ImportError:
    PYSMILE_AVAILABLE = False

from whakaaribn import moving_average


def _require_pysmile(method):
    """Decorator that raises ImportError if pysmile is not available."""
    def wrapper(*args, **kwargs):
        if not PYSMILE_AVAILABLE:
            raise ImportError(
                "pysmile and pysmile_license are required to use WhakaariSmileModel. "
                "Install them with:\n"
                "  pip install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile\n"
                "Then copy your BayesFusion license file to the site-packages directory."
            )
        return method(*args, **kwargs)
    wrapper.__doc__ = method.__doc__
    wrapper.__name__ = method.__name__
    return wrapper


class WhakaariSmileModel(BaseEstimator):
    def __init__(
        self,
        modelfile: Optional[str] = None,
        smoothing: Optional[int] = None,
        randomize: bool = False,
        uniformize: bool = False,
        nstates: int = 3,
        debug: bool = False,
        seed: Optional[int] = None,
        eq_sample_size: Optional[int] = None,
        ex_nodes: Optional[list] = None
    ):
        self.modelfile = modelfile
        self.smoothing = smoothing
        self.randomize = randomize
        self.uniformize = uniformize
        if self.randomize and self.uniformize:
            raise ValueError(
                "Can't randomize and uniformize at the same time.")
        self.nstates = nstates
        self.debug = debug
        self.seed = seed
        self.eq_sample_size = eq_sample_size
        self.ex_nodes = ex_nodes

    @_require_pysmile
    def create_network(self):
        model = pysmile.Network()
        nodes = OrderedDict(
            {
                "eruptions": 2,
                "Eqr": self.nstates,
                "CO2": self.nstates,
                "RSAM": self.nstates,
                "SO2": self.nstates,
                "H2S": self.nstates,
            }
        )
        for node in nodes.items():
            node_name, nstates = node
            node = self.add_node(model, node_name, np.arange(
                nstates), np.ones(nstates)/nstates)
        for i in range(len(nodes) - 1):
            for j in range(i + 1, len(nodes)):
                model.add_arc(list(nodes.keys())[i], list(nodes.keys())[j])
        return model

    def add_node(self, model, id, states, cpt=None):
        handle = model.add_node(pysmile.NodeType.CPT, id)
        states_count = model.get_outcome_count(handle)
        for i in range(0, states_count):
            model.set_outcome_id(handle, i, str(states[i]))
        for i in range(states_count, len(states)):
            model.add_outcome(handle, str(states[i]))
        if cpt is not None:
            model.set_node_definition(handle, cpt)
        return handle

    def set_cpt(self, model, id, cpt):
        handle = model.get_node(id)
        model.set_node_definition(handle, cpt)

    def get_cpt(self, model, id):
        handle = model.get_node(id)
        return model.get_node_definition(handle)

    def add_arc(self, model, node1, node2):
        model.add_arc(node1, node2)

    @_require_pysmile
    def fit(self, X, y):
        self.model = self.create_network()
        data_bin = X.copy()
        data_bin["eruptions"] = y
        # The following line is needed for sklearn compatibility,
        # but it is not used in the model itself
        self.classes_ = np.unique(y)
        ds = pysmile.learning.DataSet()
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        data_bin.to_csv(fname, na_rep="*", index=False)
        ds.read_file(fname)
        matching = ds.match_network(self.model)
        em = pysmile.learning.EM()
        if self.seed is not None:
            em.set_seed(self.seed)
        em.set_randomize_parameters(self.randomize)
        em.set_uniformize_parameters(self.uniformize)
        if self.eq_sample_size is not None:
            em.set_eq_sample_size(self.eq_sample_size)
        if self.ex_nodes is None:
            self.ex_nodes = []
        em.learn(ds, self.model, matching, self.ex_nodes)
        self.model.update_beliefs()
        os.remove(fname)
        if self.modelfile is not None:
            self.model.write_file(self.modelfile)

    def reset(self):
        self.model.clear_all_evidence()
        self.model.update_beliefs()

    def __str__(self):
        self.model.update_beliefs()
        msgs = []
        for nhandle in self.model.get_all_nodes():
            nid = self.model.get_node_id(nhandle)
            if self.model.is_evidence(nhandle):
                msg = "{} has evidence set to: {}"
                msg = msg.format(nid, self.model.get_evidence(nhandle))
                msgs.append(msg)
            else:
                posteriors = self.model.get_node_value(nhandle)
                for i in range(0, len(posteriors)):
                    msg = "P({}={}) = {}"
                    msg = msg.format(
                        nid, self.model.get_outcome_id(
                            nhandle, i), posteriors[i]
                    )
                    msgs.append(msg)
        return "\n".join(msgs)

    def set_evidence(self, model, node, evidence):
        if evidence is not None and not np.isnan(evidence):
            model.set_evidence(node, f"State{int(float(evidence))}")
        model.update_beliefs()

    @_require_pysmile
    def predict_proba(self, X):
        if not hasattr(self, "model"):
            if self.modelfile is not None and os.path.isfile(self.modelfile):
                self.model = pysmile.Network()
                self.model.read_file(self.modelfile)
            else:
                raise ValueError("Model not fitted or modelfile not found.")

        proba = np.ones((X.shape[0], 2))
        for r in range(X.shape[0]):
            for node_name in X.columns:
                val = X[node_name].iloc[r]
                if not val == "*":
                    try:
                        self.set_evidence(self.model, node_name, val)
                    except SMILEException as e:
                        print(r)
                        X.to_csv("SMILE_exception_training_data.csv")
                        self.model.write_file("SMILE_exception_model.xdsl")
                        raise (e)
            try:
                self.model.update_beliefs()
            except SMILEException as e:
                X.to_csv("SMILE_exception_training_data.csv")
                self.model.write("SMILE_exception_model.xdsl")
                raise (e)
            proba[r, 0] = self.model.get_node_value("eruptions")[0]
            proba[r, 1] = self.model.get_node_value("eruptions")[1]
            self.model.clear_all_evidence()
            self.model.update_beliefs()

        if self.smoothing is not None:
            proba = moving_average(
                proba, window_size=self.smoothing, axis=0, nan=False)
        return proba

    @_require_pysmile
    def from_pgmpy_model(self, pgmpy_model):
        model = pysmile.Network()
        for node_name, states in pgmpy_model.states.items():
            self.add_node(model, node_name, states)
        for edge in pgmpy_model.edges:
            nhandle_parent = model.get_node(edge[0])
            nhandle_child = model.get_node(edge[1])
            self.add_arc(model, nhandle_parent, nhandle_child)

        cpds = pgmpy_model.get_cpds()
        for cpd in cpds:
            self.set_cpt(model, cpd.variables[0], cpd.get_values().T.flatten())
        self.model = model
        if self.modelfile is not None:
            self.model.write_file(self.modelfile)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
