import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from pysmile import SMILEException
from sklearn import set_config
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from tonik import Storage

from whakaaribn import (
    BayesNet,
    Discretizer,
    convert_probability,
    load_all_whakaari_data,
    load_whakaari_catalogue,
    moving_average,
)

set_config(transform_output="pandas")


def pre_eruption_window(data: np.ndarray, ewin: int) -> np.ndarray:
    """
    Generate a time-series with pre_eruption windows.

    Parameters
    ----------
        data: numpy.ndarray
            Binary time-series with 1s indicating eruptions
        ewin: int
            Length of the pre-eruption window in days

    Returns
    -------
    numpy.ndarray
            The modified time-series with 1s in the pre-eruption windows.
    """
    data_ = data.copy()
    idx = np.where(data_ == 1)[0]
    for win_end in idx:
        win_start = max(0, win_end - ewin + 1)
        data_[win_start:win_end] = 1
    return data_


def get_group_labels(
    startdate=datetime(2010, 1, 1),
    enddate=datetime.utcnow(),
    ndays=30,
    group_names="abcdefghijklmnopqrstuvwxyz",
    min_interval=10,
    min_size=2,
):
    """
    Generate group labels to split the data set by.

    Parameters
    ----------
        startdate: Timestamp, str
            Beginning of the time-series
        enddate: Timestamp, str
            End of the time-series
        ndays: int
            Number of days after an eruption to end
            a group. If the time difference between
            two eruptions is less than ndays, the
            midpoint between the eruptions is used
            instead.
        group_names: iterable
            A sequence of group labels. The sequence
            should be at least as long as the number of
            groups.
        min_interval: int
            Minimum interval between eruptions to
            consider them separate events.
        min_size: int
            Minimum eruption size to consider.
    Returns
    -------
        list
            Group labels
    """
    eruptions = load_whakaari_catalogue(min_size, "0D")
    dfe = eruptions.loc[startdate:]
    dt = []
    group_times = []
    t_old = startdate
    for eidx in range(1, dfe.index.size):
        ie = dfe.index[eidx] - dfe.index[eidx - 1]
        if ie < np.timedelta64(min_interval, "D"):
            continue
        dt = np.timedelta64(min(ie / 2.0, np.timedelta64(ndays, "D")), "D")
        end = dfe.index[eidx - 1] + dt
        group_times.append((t_old, end))
        t_old = end
    end = dfe.index[eidx] + np.timedelta64(ndays, "D")
    group_times.append((t_old, end))
    t_old = end
    group_times.append((t_old, pd.Timestamp(enddate)))
    dates = pd.date_range(startdate, enddate, freq="1D")
    group_labels = []
    for i, start_end in enumerate(group_times):
        start, end = start_end
        group_length = len(dates[(dates >= start) & (dates < end)])
        group_labels += [group_names[i]] * group_length
    group_labels.append(group_labels[-1])
    return np.array(group_labels)


class WhakaariModel(BaseEstimator):
    def __init__(
        self,
        modelfile,
        hidden_nodes=True,
        eq_sample_size=0,
        expert_only=False,
        smoothing=None,
        randomize=False,
        uniformize=False,
        debug=False,
    ):
        self.hidden_nodes = hidden_nodes
        self.modelfile = modelfile
        self.eq_sample_size = eq_sample_size
        self.hidden_proba_ = None
        self.smoothing = smoothing
        self.randomize = randomize
        self.uniformize = uniformize
        self.debug = debug
        self.expert_only = expert_only

    def fit(self, X, y):
        y_transformed = np.where(y == 1, "yes", "no")
        data_bin = X.copy()
        data_bin = data_bin.assign(eruptions=y_transformed)
        self.classes_ = np.unique(y)
        self.net_ = BayesNet()
        if self.modelfile is not None:
            if not os.path.isfile(self.modelfile):
                raise FileNotFoundError("Can't find file " + self.modelfile)
            self.net_.net.read_file(self.modelfile)
            if isinstance(self.eq_sample_size, int):
                eq_sample_size = self.eq_sample_size
            elif isinstance(self.eq_sample_size, float):
                eq_sample_size = int(self.eq_sample_size * X.shape[0])
            else:
                raise ValueError("eq_sample_size has to be int or float.")
            try:
                if not self.expert_only:
                    self.net_.fit(
                        data_bin,
                        eq_sample_size=eq_sample_size,
                        uniformize=self.uniformize,
                        randomize=self.randomize,
                    )
            except SMILEException as e:
                data_bin.to_csv("SMILE_exception_training_data.csv", index=False)
                self.net_.write("SMILE_exception_model.xdsl")
                raise (e)

    def predict_proba(self, X):
        proba = np.ones((X.shape[0], len(self.classes_)))
        self.hidden_proba_ = np.ones((X.shape[0], 4))
        for r in range(X.shape[0]):
            for node_name in X.columns:
                val = X[node_name].iloc[r]
                if not val == "*":
                    try:
                        self.net_.net.set_evidence(node_name, str(val))
                    except SMILEException as e:
                        print(r)
                        X.to_csv("SMILE_exception_training_data.csv")
                        self.net_.write("SMILE_exception_model.xdsl")
                        raise (e)
            try:
                self.net_.net.update_beliefs()
            except SMILEException as e:
                X.to_csv("SMILE_exception_training_data.csv")
                self.net_.write("SMILE_exception_model.xdsl")
                raise (e)
            proba[r, 0] = self.net_.net.get_node_value("eruptions")[0]
            proba[r, 1] = self.net_.net.get_node_value("eruptions")[1]
            if self.hidden_nodes:
                try:
                    self.hidden_proba_[r, 0] = self.net_.net.get_node_value("magma")[0]
                    self.hidden_proba_[r, 1] = self.net_.net.get_node_value("magma")[1]
                    self.hidden_proba_[r, 2] = self.net_.net.get_node_value("seal")[0]
                    self.hidden_proba_[r, 3] = self.net_.net.get_node_value("seal")[1]
                except SMILEException as e:
                    print(r, X.iloc[r])
                    raise (e)
            self.net_.reset()
        if self.smoothing is not None:
            proba = moving_average(proba, window_size=self.smoothing, axis=0, nan=False)
            if self.hidden_nodes:
                self.hidden_proba_ = moving_average(
                    self.hidden_proba_, window_size=self.smoothing, axis=0, nan=False
                )
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y=None, weight=None):
        sensitivity_ = defaultdict(lambda: np.ones(X.shape[0]) * np.nan)
        self.net_.net.set_target("eruptions", True)
        for r in range(X.shape[0]):
            for node_name in X.columns:
                val = X[node_name].iloc[r]
                if not val == "*":
                    try:
                        self.net_.net.set_evidence(node_name, str(val))
                    except SMILEException as e:
                        X.to_csv("SMILE_exception_training_data.csv")
                        self.net_.write("SMILE_exception_model.xdsl")
                        raise (e)
            try:
                self.net_.net.update_beliefs()
            except SMILEException as e:
                X.to_csv("SMILE_exception_training_data.csv")
                self.net_.write("SMILE_exception_model.xdsl")
                raise (e)
            sens_res = self.net_.net.calc_sensitivity()
            for node_name in X.columns:
                sens = sens_res.get_node_sensitivity(node_name, "eruptions", "yes")
                sensitivity_[node_name][r] = (
                    np.mean(np.abs(np.array(sens.sensitivity))) / 2.0
                )
            self.net_.reset()
        self.net_.net.clear_all_targets()
        return sensitivity_


class SequentialGroupSplit:
    def __init__(self, groups):
        self.groups = groups

    def split(self, X, y=None, groups=None):
        _data = X.copy()
        conds = []
        group_ids = np.unique(self.groups)
        for i in range(group_ids[0:-1].size):
            conds.append(f"(self.groups == '{group_ids[i]}')")
            try:
                _data_train = _data[eval("|".join(conds))]
            except KeyError as e:
                print(conds)
                raise e
            train_idx0 = _data.index.get_indexer([_data_train.index[0]])[0]
            train_idx1 = _data.index.get_indexer([_data_train.index[-1]])[0]
            train = np.arange(train_idx0, train_idx1 + 1)
            _data_test = _data[self.groups == group_ids[i + 1]]
            test_idx0 = _data.index.get_indexer([_data_test.index[0]])[0]
            test_idx1 = _data.index.get_indexer([_data_test.index[-1]])[0]
            test = np.arange(test_idx0, test_idx1 + 1)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return np.unique(self.groups).size - 1


class WhakaariForecasts(object):
    def __init__(self, output_dir: str = tempfile.gettempdir()):
        self.forecast_store = Storage("eruption_forecasts", rootdir=output_dir)
        self.zarr_store = os.path.join(output_dir, "whakaari_forecasts.zarr")
        self.modelfile_dir = os.path.join(output_dir, "models")
        os.makedirs(self.modelfile_dir, exist_ok=True)

    def group_train_test_split(self, data, groups):
        assert len(data) == len(groups)
        data_ = pd.DataFrame(data.copy())
        data_["group"] = groups
        test_data = data_[data_["group"] == "d"]
        remainder = data_[data_["group"] == "e"]
        train_data = data_.drop(data_[data_.group == "d"].index)
        train_data = train_data.drop(train_data[data_.group == "e"].index)

        test = test_data.drop(columns=["group"])
        train = train_data.drop(columns=["group"])
        remainder = remainder.drop(columns=["group"])
        return train, test, remainder

    def get_train_test_data(self, data, ndays=30, min_interval=360, min_size=2):
        groups = get_group_labels(
            data.index[0],
            data.index[-1],
            ndays=ndays,
            min_interval=min_interval,
            min_size=min_size,
        )
        X_train, X_test, X_remainder = self.group_train_test_split(data, groups)
        eruptions = load_whakaari_catalogue(min_size, "0D")
        dfe = eruptions.loc[data.index[0] :]
        dates = pd.date_range(data.index[0], data.index[-1], freq="1D")
        dfe = dfe.reindex(dates, fill_value=0)
        dfe.drop(["delta", "tvalue"], axis=1, inplace=True)
        y_train, y_test, y_remainder = self.group_train_test_split(
            np.sign(dfe["Activity_Scale"]), groups
        )
        return X_train, X_test, X_remainder, y_train, y_test, y_remainder, groups

    def calculate_node_positions(self, num_nodes, radius, offset=(0, 0)):
        positions = []
        for i in range(num_nodes):
            theta = (2 * math.pi * i) / num_nodes
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            positions.append((int(x + offset[0]), int(y + offset[1])))
        return positions

    def fully_connected(self, nodes):
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append((nodes[i], nodes[j]))
        return edges

    def create_network(self, network_file, node_names, positions, edges):
        net_ = BayesNet()
        for node, pos in zip(node_names, positions):
            node_name, states = node
            nstates = len(states)
            node = net_.add_node(
                node_name,
                states,
                np.ones(nstates) / nstates,
                description=node_name,
                position=pos,
            )
        for parent, child in edges:
            net_.add_arc(parent, child)
        net_.write(network_file)

    def forecasts(
        self,
        data: pd.DataFrame,
        exclude_from_test: Sequence = (),
        pew: int = 30,
        expert_only: bool = False,
        eq_sample_size: int = 1,
        modelfile: str = "data/Whakaari_4s_initial1.xdsl",
        bins: tuple = (0, 5, 95, 100),
        hidden_nodes: bool = True,
        uniformize: bool = False,
        randomize: bool = False,
        recompute=False,
        save_trained_model=False,
        smoothing=None,
    ):
        """
        Compute BN forecasts
        """
        group = "/model={}/expert_only={}/bins={}/eq_sample_size={}/pew={}/exclude_from_test={}".format(
            modelfile.replace("/", "_"),
            expert_only,
            str(bins),
            eq_sample_size,
            pew,
            exclude_from_test,
        )
        path = os.path.join(self.zarr_store, group[1:])
        if os.path.isdir(path) and not recompute:
            xds = xr.open_zarr(path, consolidated=False)
            print(f"Loading forecasts from {path}")
            return xds
        data_fill = data.ffill(axis=0)
        data_fill.loc["2022-07-01":, "RSAM"] = np.nan
        data_fill.loc["2022-07-01":, "Eqr"] = np.nan
        pipe = Pipeline(
            [
                ("discretize", Discretizer(bins=bins, strategy="quantile", names=None)),
                (
                    "clf",
                    WhakaariModel(
                        expert_only=expert_only,
                        uniformize=uniformize,
                        eq_sample_size=eq_sample_size,
                        randomize=randomize,
                        hidden_nodes=hidden_nodes,
                        modelfile=modelfile,
                        smoothing=smoothing,
                    ),
                ),
            ]
        )

        x_train, x_test, x_remainder, y_train, y_test, y_remainder, groups = (
            self.get_train_test_data(data_fill, ndays=30)
        )
        y_train = pre_eruption_window(y_train, pew)
        y_test = pre_eruption_window(y_test, pew)
        y_all = pd.concat([y_train, y_test, y_remainder])
        cv = SequentialGroupSplit(groups)
        probs = np.zeros(data_fill.shape[0])
        magma = np.zeros(data_fill.shape[0])
        seal = np.zeros(data_fill.shape[0])
        sens = np.zeros(data_fill.shape)
        disc_data = np.full(data_fill.shape, "*", dtype="<U7")
        init = True
        for train, test in cv.split(data_fill):
            pipe.fit(data_fill.iloc[train], y_all.iloc[train])
            if init:
                probs[train] = pipe.predict_proba(data_fill.iloc[train])[:, 1]
                magma[train] = pipe["clf"].hidden_proba_[:, 1]
                seal[train] = pipe["clf"].hidden_proba_[:, 3]
                init = False
            _data_test = data_fill.copy()
            for col in exclude_from_test:
                _data_test[col] = np.nan
            probs[test] = pipe.predict_proba(_data_test.iloc[test])[:, 1]
            magma[test] = pipe["clf"].hidden_proba_[:, 1]
            seal[test] = pipe["clf"].hidden_proba_[:, 3]
            _sens = pd.DataFrame(pipe.score(_data_test.iloc[test]))
            sens[test, :] = _sens.values[:]
            disc_data[test, :] = pipe[:-1].transform(data_fill.iloc[test])

        if save_trained_model:
            trained_modelfile = modelfile.replace(".xdsl", "_trained.xdsl")
            pipe["clf"].net_.write(trained_modelfile)

        xds = xr.Dataset(
            {
                "probs": (["time"], probs),
                "probs_min": (["time"], probs),
                "probs_max": (["time"], probs),
                "magma": (["time"], magma),
                "magma_min": (["time"], magma),
                "magma_max": (["time"], magma),
                "seal": (["time"], seal),
                "seal_min": (["time"], seal),
                "seal_max": (["time"], seal),
                "sens": (["time", "type"], sens),
                "original_data": (["time", "type"], data.values),
                "discrete_data": (["time", "type"], disc_data),
                "y_all": (["time"], y_all.values.squeeze()),
            },
            coords={
                "time": data.index.tz_localize(None),
                "type": data.columns.astype(str),
            },
        )
        if save_trained_model:
            xds.to_zarr(self.zarr_store, group=group, mode="a")
        return xds

    def setup_ensembles(self):
        params_gcv = {
            2: [(0, 10, 100), (0, 50, 100), (0, 90, 100)],
            3: [(0, 5, 95, 100), (0, 33, 66, 100), (0, 25, 75, 100)],
            4: [
                (0, 25, 50, 75, 100),
                (0, 5, 50, 95, 100),
                (0, 10, 50, 90, 100),
                (0, 20, 50, 80, 100),
            ],
            5: [(0, 20, 40, 60, 80, 100), (0, 5, 20, 80, 95, 100)],
        }
        return params_gcv

    def ensemble_forecasts(self, pew: int = 30):
        data = load_all_whakaari_data(
            fill_method=None,
            startdate=datetime(2009, 1, 1),
            enddate=datetime.utcnow(),
            ignore_data=("LP", "VLP"),
            ignore_cache=True,
            ignore_all_caches=True,
        )
        fts = []
        for nstates in [2, 3, 4, 5]:
            obs_states = ["state_{:d}".format(i) for i in range(nstates)]
            binary_states = ["no", "yes"]
            node_names = [
                ("eruptions", binary_states),
                ("Eqr", obs_states),
                ("CO2", obs_states),
                ("RSAM", obs_states),
                ("SO2", obs_states),
                ("H2S", obs_states),
            ]
            radius = 200
            positions = self.calculate_node_positions(
                len(node_names), radius, (radius, radius)
            )
            edges = self.fully_connected([node for node, _ in node_names])
            modelfile = os.path.join(
                self.modelfile_dir, f"fully_connected_model_{nstates}_states.xdsl"
            )
            self.create_network(modelfile, node_names, positions, edges)
            for bins in self.setup_ensembles()[nstates]:
                xds = self.forecasts(
                    data,
                    pew=pew,
                    expert_only=False,
                    modelfile=modelfile,
                    bins=bins,
                    hidden_nodes=False,
                    uniformize=True,
                    randomize=False,
                    recompute=True,
                    save_trained_model=False,
                    smoothing=30,
                )

                fts.append(xds["probs"].values)

        xds_all = xr.Dataset(
            {
                "emean": (["datetime"], np.median(fts, axis=0)),
                "emin": (["datetime"], np.percentile(fts, 16, axis=0)),
                "emax": (["datetime"], np.percentile(fts, 84, axis=0)),
            },
            coords={
                "datetime": xds["probs"].time.values,
            },
        )
        xds_28 = xr.Dataset(
            {
                "emean": (
                    ["datetime"],
                    convert_probability(xds_all["emean"].values, pew, 28),
                ),
                "emin": (
                    ["datetime"],
                    convert_probability(xds_all["emin"].values, pew, 28),
                ),
                "emax": (
                    ["datetime"],
                    convert_probability(xds_all["emax"].values, pew, 28),
                ),
            },
            coords={
                "datetime": xds["probs"].time.values,
            },
        )
        xds_91 = xr.Dataset(
            {
                "emean": (
                    ["datetime"],
                    convert_probability(xds_all["emean"].values, pew, 91),
                ),
                "emin": (
                    ["datetime"],
                    convert_probability(xds_all["emin"].values, pew, 91),
                ),
                "emax": (
                    ["datetime"],
                    convert_probability(xds_all["emax"].values, pew, 91),
                ),
            },
            coords={
                "datetime": xds["probs"].time.values,
            },
        )
        st_28 = self.forecast_store.get_substore("whakaari", "28days")
        st_28.save(xds_28, mode="w")
        st_91 = self.forecast_store.get_substore("whakaari", "91days")
        st_91.save(xds_91, mode="w")
        data.index.name = "datetime"
        data_xrd = data.to_xarray()
        st_data = self.forecast_store.get_substore("whakaari", "data")
        st_data.save(data_xrd, mode="w")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Whakaari forecasts")
    parser.add_argument(
        "--pew",
        type=int,
        default=30,
        help="Pre-eruption window length in days [default: 30]",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(os.environ["HOME"], ".cache"),
        help="Directory to store models and forecasts [default: $HOME/.cache]",
    )

    args = parser.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    wf = WhakaariForecasts(args.outdir)
    wf.ensemble_forecasts(args.pew)


if __name__ == "__main__":
    main()
