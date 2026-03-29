import logging
from collections.abc import Sequence
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from aitana import whakaari
from joblib import Parallel, delayed
from sklearn import set_config
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from whakaaribn import (
    Discretizer,
    SequentialGroupSplit,
    pre_eruption_window,
)
from whakaaribn.grid_search import my_roc_auc
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
    compute_score: bool = False,
    model_class: type[WhakaariModel | WhakaariSmileModel] = WhakaariModel,
):
    """
    Compute BN forecasts.

    Parameters
    ----------
    compute_score : bool, optional
        When ``True``, the modified ROC AUC score (``my_roc_auc``) is
        computed from the forecast probabilities and stored as the scalar
        ``"score"`` variable in the returned dataset. The eruption catalogue
        is derived from ``data["eruptions"]``.
    """
    if model_class is WhakaariSmileModel and not PYSMILE_AVAILABLE:
        raise ImportError(
            "WhakaariSmileModel requires pysmile. "
            "Install it with: pip install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile"
        )
    wm = model_class(
        smoothing=smoothing, uniformize=True, nstates=len(bins) - 1, pew=pew
    )
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
    cv = SequentialGroupSplit(data_fill.group)
    y = data_fill["eruptions"]
    eruptions = whakaari.eruptions(2, "0D", end_date=data.index[-1])
    X = data_fill.drop(columns=["eruptions", "group"])
    probs = np.zeros(X.shape[0])
    disc_data = np.full(X.shape, np.nan, dtype=float)
    scores = []
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
            if compute_score:
                score = my_roc_auc(
                    pipe,
                    _data_test.iloc[test].ffill(),
                    None,
                    eruptions=eruptions,
                    use_pew=False,
                )
                scores.append(score)

    xds = xr.Dataset(
        {
            "probs": (["datetime"], probs),
            "probs_min": (["datetime"], probs),
            "probs_max": (["datetime"], probs),
            "original_data": (["datetime", "type"], X.values),
            "discrete_data": (["datetime", "type"], disc_data),
            "y": (["datetime"], y.values.squeeze()),
            "scores": (["folds"], scores),
        },
        coords={
            "datetime": X.index.tz_localize(None),
            "folds": np.arange(len(scores)),
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
    n_jobs: int = -1,
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
        n_jobs: int
            Number of parallel jobs. -1 uses all available CPUs.
        model_class: type[WhakaariModel | WhakaariSmileModel]
            Model class used for forecasting.
    """

    results = Parallel(n_jobs=n_jobs)(
        delayed(forecast)(
            data,
            pew=pew,
            bins=bins,
            smoothing=30,
            factor=factor,
            model_class=model_class,
        )
        for _ in tqdm(range(nmodels))
    )

    fts = [xds["probs"].values for xds in results]
    datetime_coord = results[0]["probs"].datetime.values
    xds_all = xr.DataArray(
        np.array(fts),
        dims=["model", "datetime"],
        coords={"model": np.arange(len(fts)), "datetime": datetime_coord},
    )
    return xds_all


def uncertainty_analysis(
    data: pd.DataFrame,
    search_results: pd.DataFrame,
    compute_score: bool = False,
    n_jobs: int = -1,
    model_class: type[WhakaariModel | WhakaariSmileModel] = WhakaariModel,
):
    """
    Compute the spread of forecasts that were tested during the grid search.

    Parameters
    ----------
        compute_score : bool, optional
            When ``True``, the ``my_roc_auc`` score is recomputed inside
            :func:`forecast` (derived from the data) and used as the
            ``model_score`` coordinate. When ``False``, the pre-stored
            ``mean_test_mod_roc_auc_no_pew`` values from *search_results* are
            used instead.
        n_jobs: int
            Number of parallel jobs. -1 uses all available CPUs.
        model_class: type[WhakaariModel | WhakaariSmileModel]
            Model class used for forecasting.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(forecast)(
            data,
            pew=row[1]["params"]["clf__pew"],
            bins=row[1]["params"]["discretize__bins"],
            smoothing=30,
            compute_score=compute_score,
            model_class=model_class,
        )
        for row in tqdm(search_results.iterrows())
    )

    fts = [xds["probs"].values for xds in results]
    scores = (
        [xds["scores"].mean() for xds in results]
        if compute_score
        else [
            row[1]["mean_test_mod_roc_auc_no_pew"] for row in search_results.iterrows()
        ]
    )
    datetime_coord = results[0]["probs"].datetime.values
    xds_all = xr.DataArray(
        np.array(fts),
        dims=["model_score", "datetime"],
        coords={"model_score": scores, "datetime": datetime_coord},
    )
    return xds_all
