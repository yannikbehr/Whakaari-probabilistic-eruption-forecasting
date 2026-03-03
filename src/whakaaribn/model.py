import os
from collections import OrderedDict
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import BIFReader, BIFWriter
from sklearn.base import BaseEstimator

from whakaaribn import moving_average


class WhakaariModel(BaseEstimator):
    def __init__(
        self,
        modelfile: Optional[str] = None,
        smoothing: Optional[int] = None,
        randomize: bool = False,
        uniformize: bool = False,
        nstates: int = 3,
        debug: bool = False,
        seed: Optional[int] = None,
        learning_method: str = "bayesian_estimation",
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
        self.learning_method = learning_method

    def create_network(self):
        cardinality = OrderedDict(
            {
                "eruptions": 2,
                "Eqr": self.nstates,
                "CO2": self.nstates,
                "RSAM": self.nstates,
                "SO2": self.nstates,
                "H2S": self.nstates,
            }
        )
        G = nx.DiGraph()
        for node_name, states in cardinality.items():
            G.add_node(node_name, states=states)
        for i in range(len(cardinality) - 1):
            for j in range(i + 1, len(cardinality)):
                G.add_edge(list(cardinality.keys())[
                           i], list(cardinality.keys())[j])
        model = DiscreteBayesianNetwork(G)
        cpds = self._init_cpds(model, cardinality)
        if len(cpds) > 0:
            model.add_cpds(*cpds.values())
            model.check_model()
            if self.modelfile is not None:
                writer = BIFWriter(model)
                writer.write(filename=self.modelfile)
        return model

    def _init_cpds(self, model, cardinality):
        cpds = {}
        for node in model.nodes():
            parents = model.get_parents(node)
            if len(parents) < 1:
                if self.uniformize:
                    cpds[node] = TabularCPD.get_uniform(
                        node, cardinality=cardinality)
                elif self.randomize:
                    cpds[node] = TabularCPD.get_random(
                        node, cardinality=cardinality, seed=self.seed
                    )

            else:
                if self.uniformize:
                    cpds[node] = TabularCPD.get_uniform(
                        node, cardinality=cardinality, evidence=parents
                    )
                elif self.randomize:
                    cpds[node] = TabularCPD.get_random(
                        node,
                        cardinality=cardinality,
                        evidence=parents,
                        seed=self.seed,
                    )
        return cpds

    def fit(self, X, y, method="bayesian_estimation"):
        if self.modelfile is not None and os.path.isfile(self.modelfile):
            reader = BIFReader(self.modelfile)
            self.model = reader.get_model(state_name_type=int)
        else:
            self.model = self.create_network()
        data_bin = X.copy()
        data_bin["eruptions"] = y
        # The following line is needed for sklearn compatibility,
        # but it is not used in the model itself
        self.classes_ = np.unique(y)
        if method == "em":
            pnet_new = DiscreteBayesianNetwork(self.model.edges())
            em_est = EM(model=pnet_new, data=data_bin)
            learned_cpds = em_est.get_parameters(
                init_cpds="uniform", apply_smoothing=True, max_iter=1000
            )
            self.model.add_cpds(*learned_cpds)
        elif method == "maximum_likelihood":
            self.model.fit(data_bin)
        elif method == "bayesian_estimation":
            try:
                self.model.fit_update(data_bin, n_prev_samples=1)
            except ValueError as e:
                print(f"Error during fit_update: {e}")
                print(data_bin.apply(lambda s: s.unique()))
                print(self.nstates)
                for node in self.model.nodes():
                    cpd = self.model.get_cpds(node)
                    print(cpd.values.shape)
                raise e

    def simulate(self, mode="discrete", **kwargs):
        model = self.create_network()
        data = model.simulate(**kwargs)
        dates = pd.date_range("2020-01-01", periods=data.shape[0], freq="1D")
        data.index = dates
        if mode == "discrete":
            return data
        elif mode == "continuous":
            data_cont = data.drop("eruptions", axis=1)
            rng = np.random.default_rng()
            bin_width = 0.8  # You can adjust this value as needed
            jitter = rng.uniform(0, bin_width, size=data_cont.shape)
            data_cont = data_cont.astype(float) * bin_width + jitter
            data_cont["eruptions"] = data["eruptions"].astype(float)
            return data_cont

    def predict_proba(self, X):
        if not hasattr(self, "model"):
            if self.modelfile is not None and os.path.isfile(self.modelfile):
                reader = BIFReader(self.modelfile)
                self.model = reader.get_model(state_name_type=int)
            else:
                raise ValueError("Model not fitted or modelfile not found.")
        proba = self.model.predict_probability(X).values
        if self.smoothing is not None:
            proba = moving_average(
                proba, window_size=self.smoothing, axis=0, nan=False)
        return proba

    def predict(self, X):
        return self.predict_proba(X)

    def __str__(self):
        parts = []
        for name in (
            "modelfile",
            "smoothing",
            "randomize",
            "uniformize",
            "nstates",
            "debug",
            "seed",
            "learning_method",
        ):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value!r}")
        if parts:
            return f"WhakaariModel({', '.join(parts)})"
        return "WhakaariModel()"
