import logging
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from sklearn import set_config
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from whakaaribn import (
    Discretizer,
    SequentialGroupSplit,
    pre_eruption_window,
)
from whakaaribn.model import WhakaariModel
from whakaaribn.smile_model import PYSMILE_AVAILABLE, WhakaariSmileModel

set_config(transform_output="pandas")

logger = logging.getLogger(__name__)


def latest_data_point(data):
    """
    Get the latest data point in the dataset that is not null. This
    only takes into account gas flux data at the moment as other data
    is currently unavailable.
    """
    latest_valid_index = data["SO2"].last_valid_index()
    for col in ["SO2", "CO2", "H2S"]:
        last_valid_index = data[col].last_valid_index()
        if last_valid_index > latest_valid_index:
            latest_valid_index = last_valid_index
    return latest_valid_index


def forecast(
    data: pd.DataFrame,
    exclude_from_test: Sequence = (),
    pew: int = 30,
    eq_sample_size: int = 1,
    bins: tuple = (0, 5, 95, 100),
    smoothing: Optional[int] = None,
    factor: float = 0.0,
    hindcast: bool = False,
    model_class: type[WhakaariModel | WhakaariSmileModel] = WhakaariModel,
):
    """
    Compute BN forecasts
    """
    if model_class is WhakaariSmileModel and not PYSMILE_AVAILABLE:
        raise ImportError(
            "WhakaariSmileModel requires pysmile. "
            "Install it with: pip install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile"
        )
    wm = model_class(smoothing=smoothing, uniformize=True,
                     nstates=len(bins) - 1)
    data_fill = data.ffill(axis=0)
    data_fill.loc["2022-07-01":, "RSAM"] = np.nan
    data_fill.loc["2022-07-01":, "Eqr"] = np.nan
    pipe = Pipeline(
        [
            (
                "discretize",
                Discretizer(bins=bins, strategy="quantile", factor=factor),
            ),
            ("clf", wm),
        ]
    )
    y = pre_eruption_window(data_fill["eruptions"], pew)
    cv = SequentialGroupSplit(data_fill.group)
    X = data_fill.drop(columns=["eruptions", "group"])
    probs = np.zeros(X.shape[0])
    disc_data = np.full(X.shape, np.nan, dtype=float)

    if hindcast:
        pipe.fit(X, y)
        _data_test = X.copy()
        for col in exclude_from_test:
            _data_test[col] = np.nan
        probs = pipe.predict_proba(_data_test)[:, 1]
    else:
        init = True
        for train, test in cv.split(X):
            pipe.fit(X.iloc[train], y.iloc[train])
            if init:
                probs[train] = pipe.predict_proba(X.iloc[train])[:, 1]
                init = False
            _data_test = X.copy()
            for col in exclude_from_test:
                _data_test[col] = np.nan
            probs[test] = pipe.predict_proba(_data_test.iloc[test])[:, 1]
            disc_data[test, :] = pipe[:-1].transform(X.iloc[test])

    xds = xr.Dataset(
        {
            "probs": (["datetime"], probs),
            "probs_min": (["datetime"], probs),
            "probs_max": (["datetime"], probs),
            "original_data": (["datetime", "type"], X.values),
            "discrete_data": (["datetime", "type"], disc_data),
            "y": (["datetime"], y.values.squeeze()),
        },
        coords={
            "datetime": X.index.tz_localize(None),
            "type": X.columns.astype(str),
        },
    )
    return xds


def sensitivity_analysis(
    data: pd.DataFrame,
    pew: int = 30,
    bins: tuple = (0, 5, 95, 100),
    factor: float = 0.1,
    nmodels: int = 100,
    model_class: type[WhakaariModel | WhakaariSmileModel] = WhakaariModel,
):
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
        model_class: type[WhakaariModel | WhakaariSmileModel]
            Model class used for forecasting.
    """
    fts = []
    for i in tqdm(range(nmodels)):
        xds = forecast(
            data,
            pew=pew,
            bins=bins,
            smoothing=30,
            factor=factor,
            model_class=model_class,
        )
        fts.append(xds["probs"].values)
    xds_all = xr.DataArray(
        np.array(fts),
        dims=["model", "datetime"],
        coords={"model": np.arange(
            len(fts)), "datetime": xds["probs"].datetime},
    )
    return xds_all


def uncertainty_analysis(
    data: pd.DataFrame,
    search_results: pd.DataFrame,
    model_class: type[WhakaariModel | WhakaariSmileModel] = WhakaariModel,
):
    """
    Compute the spread of forecasts that were tested during the grid search.

    Parameters
    ----------
        model_class: type[WhakaariModel | WhakaariSmileModel]
            Model class used for forecasting.
    """

    pews = search_results.index.get_level_values(0).unique()
    nstates = search_results.index.get_level_values(1).unique()
    fts = []
    scores = []
    for _nstates in tqdm(nstates):
        for pew in tqdm(pews):
            for _c in search_results.loc[(pew, _nstates)].iterrows():
                _e = _c[1].params
                xds = forecast(
                    data,
                    pew=pew,
                    bins=_e["discretize__bins"],
                    smoothing=30,
                    model_class=model_class,
                )
                fts.append(xds["probs"].values)
                scores.append(_c[1].mean_test_mod_roc_auc_no_pew)
    xds_all = xr.DataArray(
        np.array(fts),
        dims=["model_score", "datetime"],
        coords={"model_score": scores, "datetime": xds["probs"].datetime},
    )
    return xds_all
