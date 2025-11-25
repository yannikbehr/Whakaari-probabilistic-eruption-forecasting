import math
import os
from typing import Optional
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
import logging

import numpy as np
import pandas as pd
import xarray as xr
from pysmile import SMILEException
from sklearn import set_config
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from tonik import Storage
from aitana import whakaari

from whakaaribn import (
    BayesNet,
    Discretizer,
    convert_probability,
    moving_average,
    get_data,
    circular_node_positions,
    fully_connected,
    create_network
)

set_config(transform_output="pandas")

logger = logging.getLogger(__name__)


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
    eruptions,
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


def group_train_test_split(data, groups):
    assert len(data) == len(groups)
    data_ = pd.DataFrame(data.copy())
    data_["group"] = groups
    test_data = data_[data_["group"] == "d"]
    remainder = data_[data_["group"] == "e"]
    train_data = data_.drop(data_[(data_.group == "d") | (data_.group == "e")].index)
    test = test_data.drop(columns=["group"])
    train = train_data.drop(columns=["group"])
    remainder = remainder.drop(columns=["group"])
    return train, test, remainder


class WhakaariModel(BaseEstimator):
    def __init__(
        self,
        modelfile,
        modeldir=None,
        hidden_nodes=True,
        eq_sample_size=0,
        expert_only=False,
        smoothing=None,
        randomize=False,
        uniformize=False,
        debug=False
    ):
        self.hidden_nodes = hidden_nodes
        self.modelfile = modelfile
        self.modeldir = modeldir
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
            modelfile = self.modelfile
            if self.modeldir is not None:
                modelfile = os.path.join(self.modeldir, modelfile)
            if not os.path.isfile(modelfile):
                raise FileNotFoundError("Can't find file " + modelfile)
            self.net_.net.read_file(modelfile)
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
    def __init__(self, start_date=datetime(2009, 1, 1), end_date=datetime(2025, 4, 16),
                 output_dir: Optional[str] = None, posteruption_days=30,
                 min_eruption_size=2, min_intereruption_int=360):
        self.output_dir = output_dir
        if output_dir is not None:
            self.zarr_store = os.path.join(output_dir, "whakaari_forecasts.zarr")
            self.modelfile_dir = os.path.join(output_dir, "models")
            os.makedirs(self.modelfile_dir, exist_ok=True)
        self.data = whakaari.load_all(
            fill_method=None,
            start_date=start_date,
            end_date=end_date,
            ignore_data=("LP", "VLP"),
            fuse_so2=False
        )
        self.eruptions = whakaari.eruptions(min_eruption_size, "0D", end_date=end_date)
        self.groups = get_group_labels(
            self.eruptions,
            self.data.index[0],
            self.data.index[-1],
            ndays=posteruption_days,
            min_interval=min_intereruption_int,
            min_size=min_eruption_size,
        )

    def latest_data_point(self):
        """
        Get the latest data point in the dataset that is not null. This
        only takes into account gas flux data at the moment as other data
        is currently unavailable.
        """
        latest_valid_index = self.data['SO2'].last_valid_index()
        for col in ['SO2', 'CO2', 'H2S']:
            last_valid_index = self.data[col].last_valid_index()
            if last_valid_index > latest_valid_index:
                latest_valid_index = last_valid_index
        return latest_valid_index


    def get_train_test_data(self, data=None):
        if data is None:
            data = self.data
        X_train, X_test, X_remainder = group_train_test_split(data, self.groups)
        dfe = self.eruptions.loc[data.index[0] :]
        dates = pd.date_range(data.index[0], data.index[-1], freq="1D")
        dfe = dfe.reindex(dates, fill_value=0)
        y_train, y_test, y_remainder = group_train_test_split(
            np.sign(dfe["Activity_Scale"]), self.groups
        )
        return X_train, X_test, X_remainder, y_train, y_test, y_remainder

    def forecasts(
        self,
        exclude_from_test: Sequence = (),
        pew: int = 30,
        expert_only: bool = False,
        eq_sample_size: int = 1,
        modelfile: str = "Whakaari_4s_initial1.xdsl",
        modeldir: str = 'data',
        bins: tuple = (0, 5, 95, 100),
        hidden_nodes: bool = True,
        uniformize: bool = False,
        randomize: bool = False,
        recompute: bool = False,
        smoothing: int = None,
        factor: float = 0.,
        hindcast: bool = False
    ):
        """
        Compute BN forecasts
        """
        if self.output_dir is not None:
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
        data_fill = self.data.ffill(axis=0)
        data_fill.loc["2022-07-01":, "RSAM"] = np.nan
        data_fill.loc["2022-07-01":, "Eqr"] = np.nan
        pipe = Pipeline(
            [
                ("discretize", Discretizer(bins=bins, strategy="quantile",
                                           names=None, factor=factor)),
                (
                    "clf",
                    WhakaariModel(
                        expert_only=expert_only,
                        uniformize=uniformize,
                        eq_sample_size=eq_sample_size,
                        randomize=randomize,
                        hidden_nodes=hidden_nodes,
                        modelfile=modelfile,
                        modeldir=modeldir,
                        smoothing=smoothing,
                    ),
                ),
            ]
        )

        x_train, x_test, x_remainder, y_train, y_test, y_remainder = (
            self.get_train_test_data(data_fill)
        )
        y_train = pre_eruption_window(y_train, pew)
        y_test = pre_eruption_window(y_test, pew)
        y_all = pd.concat([y_train, y_test, y_remainder])
        cv = SequentialGroupSplit(self.groups)
        probs = np.zeros(data_fill.shape[0])
        magma = np.zeros(data_fill.shape[0])
        seal = np.zeros(data_fill.shape[0])
        sens = np.zeros(data_fill.shape)
        disc_data = np.full(data_fill.shape, "*", dtype="<U7")

        if hindcast:
            pipe.fit(data_fill, y_all)
            probs = pipe.predict_proba(data_fill)[:, 1]
        else:
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

        if self.output_dir is not None:
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
                "original_data": (["time", "type"], data_fill.values),
                "discrete_data": (["time", "type"], disc_data),
                "y_all": (["time"], y_all.values.squeeze()),
            },
            coords={
                "time": data_fill.index.tz_localize(None),
                "type": data_fill.columns.astype(str),
            },
        )
        if self.output_dir is not None:
            xds.to_zarr(self.zarr_store, group=group, mode="a")
        return xds

    def get_best_estimator(self, search_results: dict=None):
        if search_results is None:
            grid_search_results_file = get_data('data/grid_search_results.csv')
            logger.info(f"Loading search results from {grid_search_results_file}")
            search_results = pd.read_csv(grid_search_results_file,
                                        index_col=(0, 1, 2), converters={'params': eval})
        best_pew, best_ns, best_params = search_results['mean_test_mod_roc_auc_no_pew'].idxmax()
        best_estimator = search_results.loc[(best_pew, best_ns, best_params)].params
        return best_estimator, best_pew

    def get_best_forecast(self, search_results=None, exclude_from_test=(), hindcast=False):
        best_estimator, best_pew = self.get_best_estimator(search_results)
        xds = self.forecasts(pew=best_pew, expert_only=False, exclude_from_test=exclude_from_test,
                        modelfile=best_estimator['clf__modelfile'],
                        modeldir=get_data('data'),
                        bins=best_estimator['discretize__bins'],
                        hidden_nodes=False,
                        uniformize=True,
                        randomize=False, recompute=True,
                        smoothing=30, hindcast=hindcast)
        return xds

    def sensitivity_analysis(self, factor: float=0.1, nmodels: int=100):
        """
        Compute sensitivity analysis by varying bin boundaries.

        Parameters
        ----------
            factor: float
                Factor to vary bin boundaries by. The new bin boundary will
                be sampled from a uniform distribution between (1 - factor) * old_boundary
                and (1 + factor) * old_boundary.
            nmodels: int
                Number of times to repeat sampling.
        """
        best_estimator, best_pew = self.get_best_estimator()
        fts = []
        for i in tqdm(range(nmodels)):
            xds = self.forecasts(pew=best_pew,
                            expert_only=False, exclude_from_test=(),
                            modelfile=best_estimator['clf__modelfile'],
                            modeldir=get_data('data'),
                            bins=best_estimator['discretize__bins'],
                            hidden_nodes=False,
                            uniformize=True,
                            randomize=False, recompute=True,
                            smoothing=30, factor=0.1)
            fts.append(xds['probs'].values)
        xds_all = xr.DataArray(np.array(fts), dims=['model', 'time'], coords={'model': np.arange(len(fts)), 'time': xds['probs'].time})
        return xds_all

    def uncertainty_analysis(self):
        """
        Compute the spread of forecasts that were tested during the grid search.
        """
        search_results = pd.read_csv(get_data('data/grid_search_results.csv'),
                                     index_col=(0, 1, 2), converters={'params': eval})
        pews = search_results.index.get_level_values(0).unique()
        nstates = search_results.index.get_level_values(1).unique()
        fts = []
        scores = []
        for _nstates in tqdm(nstates):
            for pew in tqdm(pews):
                for _c in search_results.loc[(pew, _nstates)].iterrows():
                    _e = _c[1].params
                    xds = self.forecasts(pew=pew, expert_only=False,
                                    modelfile=_e['clf__modelfile'],
                                    modeldir=get_data('data'),
                                    bins=_e['discretize__bins'],
                                    hidden_nodes=False,
                                    uniformize=True,
                                    randomize=False, 
                                    recompute=True,
                                    smoothing=30)

                    fts.append(xds['probs'].values)
                    scores.append(_c[1].mean_test_mod_roc_auc_no_pew)
        xds_all = xr.DataArray(np.array(fts), dims=['model_score', 'time'], coords={'model_score': scores, 'time': xds['probs'].time})
        return xds_all

def create_networks(outputdir):
    for nstates in range(2, 6):
        fout = os.path.join(outputdir, f"fully_connected_model_{nstates}_states.xdsl")
        if os.path.isfile(fout):
            continue
        obs_states = ["state_{:d}".format(i) for i in range(nstates)]
        binary_states = ["no", "yes"]
        node_names = [('eruptions', binary_states), ('Eqr', obs_states),
                    ('CO2', obs_states), ('RSAM', obs_states),
                    ('SO2', obs_states), ('H2S', obs_states)]
        positions = circular_node_positions(len(node_names)) 
        edges = fully_connected([node for node, _ in node_names])
        create_network(fout, node_names, positions, edges) 

    
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Whakaari forecasts")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(os.environ["HOME"], ".cache"),
        help="Directory to store models and forecasts [default: $HOME/.cache]",
    )
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--ensemble", action="store_true", help="Run ensemble forecasts.")
    parser.add_argument("--recompute", action="store_true", help="Recompute results even if they are up-to-date.")
    args = parser.parse_args(argv) 
    os.makedirs(args.outdir, exist_ok=True)
    create_networks(get_data('data'))

    end_date = datetime.now(tz=timezone.utc).date()
    wf = WhakaariForecasts(start_date=datetime(2009, 1, 1), end_date=end_date)
    latest_data_point = wf.latest_data_point()
    forecast_store = Storage('whakaari_forecasts', args.outdir)
    try:
        latest_update = forecast_store('best_model', metadata=True)['update_log'].values[-1]
        latest_update = pd.Timestamp(latest_update)
    except KeyError:
        latest_update = pd.Timestamp(2009, 1, 1, tz='UTC') 

    if latest_data_point > latest_update or args.recompute:
        logger.info("Running forecasts")
        datasets = {}
        xds_best = wf.get_best_forecast()
        datasets['best_model'] = (["datetime"], xds_best['probs'].data)
        if args.sensitivity:
            xds_sensitivity = wf.sensitivity_analysis()
            median_sens = xds_sensitivity.median(dim='model')
            min_sens = xds_sensitivity.chunk(dict(model=-1)).min('model')
            max_sens = xds_sensitivity.chunk(dict(model=-1)).max('model')
            datasets['median_sensitivity'] = (["datetime"], median_sens.data)
            datasets['max_sensitivity'] = (["datetime"], max_sens.data)
            datasets['min_sensitivity'] = (["datetime"], min_sens.data)
        if args.ensemble:
            xds_ensemble = wf.uncertainty_analysis()
            median_ens = xds_ensemble.median('model_score')
            min_ens = xds_ensemble.chunk(dict(model_score=-1)).min('model_score')
            max_ens = xds_ensemble.chunk(dict(model_score=-1)).max('model_score')
            datasets['median_ensemble'] = (["datetime"], median_ens.data)
            datasets['max_ensemble'] = (["datetime"], max_ens.data)
            datasets['min_ensemble'] = (["datetime"], min_ens.data)
        xds = xr.Dataset(datasets, coords={"datetime": xds_best.time})
        xds.attrs['latest_update'] = str(latest_data_point)
        forecast_store.save(xds)
        output_data = wf.data.copy()
        output_data.index.name = 'datetime'
        output_data.index = output_data.index.tz_localize(None)
        forecast_store.save(output_data.to_xarray())


if __name__ == "__main__":
    main()
