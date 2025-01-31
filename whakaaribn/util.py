import hashlib
import io
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.datasets import get_data_home
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm

from whakaaribn.assimilate import LocalLinearTrend


def convert_probability(prob: np.ndarray, hin: int, hnew: int) -> np.ndarray:
    """
    Convert probability between different forecast horizons
    assuming a binomial distribution.

    Parameters
    ----------
    prob: np.ndarray
          Probabilities with forecast horizon hin.
    hin: int
         Forecast horizon of input (in days).
    hnew: int
          New forecast horizon (in days).

    Returns
    -------
    np.ndarray
         Converted probabilities.

    """
    p1D = 1 - (1 - prob) ** (1 / hin)
    return 1 - (1 - p1D) ** hnew


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Compute a hash for a pandas dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
         Input dataframe to hash.
    Returns
    -------
    str
        Hash of the dataframe.
    """

    # Hash the DataFrame
    hash_series = pd.util.hash_pandas_object(df)
    hash_string = hash_series.astype(str).values.sum()
    hash_value = hashlib.sha256(hash_string.encode()).hexdigest()
    return hash_value


def gradient(df, period="14D"):
    """
    Compute gradient for time series by first smoothing
    the time series and then computing the first-order difference.

    Parameters:
    -----------
        :param df: Dataframe
        :type df: :class:`~pandas.DataFrame`
        :param period: Period over which to compute a rolling mean.
        :type period: str
    """
    df_smooth = df.rolling(period).mean()
    df_grad = df_smooth.diff()
    df_grad.obs.iloc[0] = 0.0
    return df_grad


def bin_data(
    data,
    bins,
    strategy="quantile",
    returnbins=False,
    dropzeros=True,
    limits=None,
    names=None,
    factor=0,
    seed=None,
):
    """
    Bin a 1D array such that either the
    same number of observations fall within each bin or that
    the bins have equal width.

    :param data: Dataset to be binned.
    :type data: :class:`numpy.ndarray`
    :param bins: Number of required bins or bin edges.
    :type bins: sequence or int
    :param strategy: Type of binning. Can be either 'quantile' for
                     equal frequency or 'uniform' for equal size
                     bins.
    :type strategy: str
    :param returnbins: If 'True' return the bin boundaries.
    :type returnbins: bool
    :param dropzeros: Only use data greater than zero.
    :type dropzeros: bool
    :param limits: Provide limits to the bins.
    :type limits: sequence
    :param names: Bin names.
    :type names: sequence
    :param factor: Factor between 0 and 1 by which to vary bin
                   boundaries.
    :type factor: float
    :param seed: Random seed for bin boundary variation.
    :type seed: int
    :returns: A dataframe with a new column containing the bin
              that every value is assigned to. Optionally also
              returns the bins themselves.
    :rtype: :class:`pandas.DataFrame`
    """
    _data = data.copy()
    _data = _data[~np.isnan(_data)]
    if dropzeros:
        _data = _data[_data > 0]
    if limits is not None:
        lmin, lmax = limits
        _data = _data[(_data >= lmin) & (_data <= lmax)]

    dmin = _data.min()
    dmax = _data.max()
    _bins = []
    if strategy == "quantile":
        if np.ndim(bins) == 0:
            percentiles = np.linspace(0, 1, bins + 1) * 100.0
        elif np.ndim(bins) == 1:
            percentiles = bins
        for _p in percentiles:
            _bins.append(np.percentile(_data, _p))
    elif strategy == "uniform":
        if np.ndim(bins) == 0:
            _bins = np.linspace(dmin, dmax, bins + 1)
        elif np.ndim(bins) == 1:
            _bins = bins

    # now vary bin boundaries to assess uncertainty
    _tbins = []
    rs = np.random.default_rng(seed)
    for _b in _bins:
        bin_min = (1 - factor) * _b
        bin_max = (1 + factor) * _b
        _tbins.append(rs.uniform(min(bin_min, bin_max), max(bin_min, bin_max)))
    # make sure new boundaries are in ascending order
    _tbins.sort()
    _bins = _tbins

    if names is None:
        bin_names = []
        for i in range(len(_bins) - 1):
            bin_names.append("state_{:d}".format(i))
    else:
        bin_names = names
    bin_names = np.array(bin_names)
    idx = np.digitize(_data, _bins, right=True)
    idx = np.where(idx == 0, 1, idx)
    idx = np.where(idx == len(_bins), len(_bins) - 1, idx)
    if returnbins:
        return bin_names[idx - 1], _bins
    return bin_names[idx - 1]


class Bin(object):
    """
    Class to define bins and query them.

    :param bins: Bin boundaries
    :type bins: sequence
    :param names: Bin names.
    :type names: sequence
    :param unit: Data units.
    :type unit: str
    :param factor: Factor between 0 and 1 by which to vary bin
                   boundaries.
    :type factor: float
    :param seed: Seed for random number generator. Only relevant if
                 factor > 0.
    :type seed: int
    """

    def __init__(self, bins, names=None, unit="", factor=0, seed=None):
        # determine whether bins are increasing or decreasing
        increasing = np.all(np.diff(bins) > 0)
        # now vary bin boundaries to assess uncertainty
        _tbins = []
        rs = np.random.default_rng(seed)
        for _b in bins:
            bin_min = (1 - factor) * _b
            bin_max = (1 + factor) * _b
            _tbins.append(rs.uniform(min(bin_min, bin_max), max(bin_min, bin_max)))
        # make sure new boundaries are in the right order
        _tbins.sort()
        if not increasing:
            _tbins = _tbins[::-1]
        self._bins = _tbins
        self._bin_names = names
        self.unit = unit
        self._nbins = len(self._bin_names)

    def query(self, val, extrapolate=True, nanstr="*"):
        """
        Return the bin name of the given value.

        :param val: Data value.
        :type val: float
        :param extrapolate: If 'True', assign lowest or
                            highest bin to values outside
                            the range.
        :type extrapolate: bool
        :returns: Bin name of the value.
        :rtype: str
        """
        bin_idx = np.digitize(val, self._bins, right=True)
        bin_idx -= 1
        if np.any(bin_idx < 0) or np.any(bin_idx > self._nbins - 1):
            if extrapolate:
                bin_idx = np.where(bin_idx < 0, 0, bin_idx)
                bin_idx = np.where(bin_idx > self._nbins - 1, self._nbins - 1, bin_idx)
            else:
                msg = "Value outside of range of bins: {}"
                raise ValueError(msg.format(self._bins))
        bin_names = np.array(self._bin_names)[bin_idx]
        return np.where(np.isnan(val), nanstr, bin_names)

    @property
    def bins(self):
        return self._bins

    @property
    def binnames(self):
        return self._bin_names

    def __str__(self):
        desc = ""
        for i, bn in enumerate(self.binnames):
            desc += "{:g} < {} <= {:g} {}\n".format(
                self.bins[i], bn, self.bins[i + 1], self.unit
            )
        return desc


class BinData(Bin):
    """
    Bin data, compute marginals, and query the bins with new values.

    :param data: Dataset to be binned.
    :type data: :class:`pandas.Dataframe`
    :param column: Column name of the data that needs to be
                   binned.
    :type column: str
    :param bins: Number of required bins or bin edges.
    :type bins: sequence or int
    :param btype: Type of binning. Can be either 'freq' for
                  equal frequency or 'size' for equal size
                  bins.
    :type btype: str
    :param names: Bin names.
    :type names: list
    :param dropzeros: Only use data greater than zero.
    :type dropzeros: bool
    :param limits: Provide limits to the bins before binning it.
    :type limits: sequence
    :param unit: Data units.
    :type unit: str
    :param factor: Factor between 0 and 1 by which to vary bin
                   boundaries.
    :type factor: float
    :param seed: Seed for random number generator. Only relevant if
                 factor > 0.
    :type seed: int
    """

    def __init__(
        self,
        data,
        column,
        bins,
        btype="freq",
        dropzeros=True,
        limits=None,
        names=None,
        unit="",
        factor=0,
        seed=None,
    ):
        self.column = column
        self.dropzeros = dropzeros
        self.unit = unit
        if btype == "freq":
            strategy = "quantile"
        elif btype == "size":
            strategy = "uniform"
        self.binned_data, self._bins = bin_data(
            data[column].values,
            bins,
            strategy=strategy,
            returnbins=True,
            dropzeros=dropzeros,
            limits=limits,
            names=names,
            factor=factor,
            seed=seed,
        )
        if names is None:
            self._bin_names = sorted(list(np.unique(self.binned_data)))
        else:
            self._bin_names = names
        self._nbins = len(self._bin_names)

        # Code for back compatibility with bayesnet.data_table
        _data = data[column].copy()
        _data.dropna(inplace=True)
        if dropzeros:
            _data = _data[_data > 0]
        if limits is not None:
            lmin, lmax = limits
            _data = _data[(_data >= lmin) & (_data <= lmax)]
        self._df_binned = pd.DataFrame({"obs": _data, "bin": self.binned_data})

    @property
    def dates(self):
        return self._df_binned.index

    @property
    def data(self):
        return self._df_binned

    def marginals(self, small_prob=1e-1):
        ntotal = self.binned_data.shape[0]
        unique, counts = np.unique(self.binned_data, return_counts=True)
        vals = counts / ntotal
        rvals = []
        for _bn in self.binnames:
            try:
                idx = np.where(unique == _bn)[0]
                rvals.append(float(vals[idx]))
            except TypeError:
                rvals.append(small_prob)
        # normalise so probabilities sum to one
        rvals = np.array(rvals)
        rvals /= rvals.sum()
        return list(rvals)


class Discretizer(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A sklearn compatible binning class.

    >>> desc = Discretizer(bins=(0, 5, 95, 100))
    >>> desc.fit_transform(np.tile(np.arange(5)[:, np.newaxis], (1,3)))
    array([['low', 'low', 'low'],
           ['medium', 'medium', 'medium'],
           ['medium', 'medium', 'medium'],
           ['medium', 'medium', 'medium'],
           ['high', 'high', 'high']], dtype='<U6')
    >>> desc.transform(np.tile(np.ones(3)[:, np.newaxis], (1, 3)))
    array([['medium', 'medium', 'medium'],
           ['medium', 'medium', 'medium'],
           ['medium', 'medium', 'medium']], dtype='<U6')
    >>> desc = Discretizer(bins=(0, 5, 95, 100), extrapolate=False)
    >>> desc.fit_transform(np.tile(np.arange(5)[:, np.newaxis], (1,3)))
    array([['*', '*', '*'],
           ['medium', 'medium', 'medium'],
           ['medium', 'medium', 'medium'],
           ['medium', 'medium', 'medium'],
           ['high', 'high', 'high']], dtype='<U6')
    """

    def __init__(
        self,
        bins=(0, 33, 66, 100),
        names=("low", "medium", "high"),
        strategy="quantile",
        dropzeros=False,
        extrapolate=True,
        factor=0,
    ):
        self.bins = bins
        self.names = names
        self.strategy = strategy
        self.dropzeros = dropzeros
        self.extrapolate = extrapolate
        self.factor = factor

    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True, ensure_all_finite="allow-nan")
        try:
            self.nbins = len(self.bins) - 1
        except TypeError:
            self.nbins = self.bins
        if self.names is None:
            self.names = ["state_{:d}".format(i) for i in range(self.nbins)]
        self.bin_edges_ = np.zeros((X.shape[1], self.nbins + 1))
        for col_idx in range(X.shape[1]):
            # preserve NaN indices
            valid_idx = np.where((~np.isnan(X[:, col_idx])) & (X[:, col_idx] > 0))
            _, self.bin_edges_[col_idx] = bin_data(
                X[:, col_idx],
                self.bins,
                strategy=self.strategy,
                names=self.names,
                returnbins=True,
                dropzeros=self.dropzeros,
                factor=self.factor
            )
        self.n_features_ = X.shape[1]
        return self

    def query(self, fitted_bins, val, extrapolate=True, nanstr="*"):
        """
        Return the bin name of the given value.

        :param val: Data value.
        :type val: float
        :param extrapolate: If 'True', assign lowest or
                            highest bin to values outside
                            the range.
        :type extrapolate: bool
        :returns: Bin name of the value.
        :rtype: str
        """
        bin_idx = np.digitize(val, fitted_bins, right=True)
        bin_idx -= 1
        if np.any(bin_idx < 0) or np.any(bin_idx > self.nbins - 1):
            if extrapolate:
                bin_idx = np.where(bin_idx < 0, 0, bin_idx)
                bin_idx = np.where(bin_idx > self.nbins - 1, self.nbins - 1, bin_idx)
            else:
                bin_idx = np.where(bin_idx < 0, self.nbins, bin_idx)
                bin_idx = np.where(bin_idx > self.nbins - 1, self.nbins, bin_idx)
        bin_names = np.array(list(self.names) + [nanstr])[bin_idx]
        retval = np.where(np.isnan(val), nanstr, bin_names)
        return retval

    def transform(self, X):
        check_is_fitted(self, "n_features_")
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        X_binned = np.full(X.shape, "*", "<U8")
        for col_idx in range(X.shape[1]):
            X_binned[:, col_idx] = self.query(
                self.bin_edges_[col_idx], X[:, col_idx], extrapolate=self.extrapolate
            )
        return X_binned


class ForwardImputer(TransformerMixin, BaseEstimator):
    """
    >>> imp = ForwardImputer()
    >>> imp.fit_transform(np.array([[np.nan, 1, 2],[2, np.nan, 1], [np.nan, 5, 3]]))
    array([[nan,  1.,  2.],
           [ 2.,  1.,  1.],
           [ 2.,  5.,  3.]])
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = check_array(X, force_all_finite="allow-nan")
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_")
        X = check_array(X, force_all_finite="allow-nan")
        X_filled = np.full(X.shape, np.nan)
        for col_idx in range(X.shape[1]):
            arr = X[:, col_idx]
            prev = np.arange(len(arr))
            prev[np.isnan(arr)] = 0
            prev = np.maximum.accumulate(prev)
            X_filled[:, col_idx] = arr[prev]
        return X_filled


class ForecastImputer(TransformerMixin, BaseEstimator):
    def __init__(
        self, sigmas, nan_limit=365, kurtosis_limit=100, fill_remaining=False, new=False
    ):
        self.sigmas = sigmas
        self.nan_limit = nan_limit
        self.kurtosis_limit = kurtosis_limit
        self.fill_remaining = fill_remaining
        self.new = new

    def fit(self, X, y=None):
        X = check_array(X, force_all_finite="allow-nan")
        self.n_features_ = X.shape[1]
        # create a unique filename for the imputer
        # that is based on the input data
        output = io.BytesIO()
        np.save(output, X)
        hash = hashlib.sha256(output.getvalue()).hexdigest()
        self.filename = os.path.join(get_data_home(), f"forecast_imputer_{hash}.npy")
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_")
        if not self.new:
            try:
                X_filled = np.load(self.filename)
            except FileNotFoundError:
                pass
        X = check_array(X, force_all_finite="allow-nan")
        X_filled = np.ones((X.shape)) * np.nan
        dates = pd.date_range("1970-01-01", freq="D", periods=X.shape[0])
        for colidx in tqdm(range(X.shape[1])):
            _data = pd.Series(X[:, colidx], index=dates)
            if _data.kurtosis() > self.kurtosis_limit:
                if self.fill_remaining:
                    _data.fillna(_data.median(), inplace=True)
                X_filled[:, colidx] = _data.values
                continue
            _data_fill = _data.copy()
            _data_valid = _data[~_data.isnull()]
            _date_first = _data_valid.index[0]
            initial = True
            for i in range(5, _data_valid.shape[0] - 1):
                _date = _data_valid.index[i]
                # find forecast horizon
                idx = np.min(
                    np.where((_data_valid.index - _date) > pd.Timedelta(days=0))
                )
                idx_end = np.where(_data.index == _data_valid.index[idx])[0]
                idx_start = np.where(_data.index == _date)[0]
                fh_start = _data.index[idx_start + 1][0]
                fh_end = _data.index[idx_end - 1][0]
                if (idx_end - idx_start) < 2:
                    continue
                _mod = LocalLinearTrend(_data.loc[_date_first:_date])
                try:
                    with _mod.fix_params({"sigma2.measurement": self.sigmas[colidx]}):
                        _res = _mod.fit(disp=False, maxiter=200)
                except ConvergenceWarning:
                    try:
                        _res = _mod.fit(disp=False, maxiter=200)
                    except ConvergenceWarning as e:
                        print(e)
                if initial:
                    prediction = pd.Series(
                        _res.smoothed_state[0, :],
                        index=_data.loc[_date_first:_date].index,
                    )
                    invalid_dates = _data.loc[_date_first:_date][
                        _data.loc[_date_first:_date].isnull()
                    ].index
                    _data_fill.loc[invalid_dates] = prediction.loc[invalid_dates]
                    initial = False
                forecast = _res.get_forecast(int(idx_end - idx_start))
                _data_fill.loc[fh_start:fh_end] = forecast.predicted_mean.loc[
                    fh_start:fh_end
                ]
            # final
            _mod = LocalLinearTrend(_data.loc[: _data_valid.index[-1]])
            try:
                with _mod.fix_params({"sigma2.measurement": self.sigmas[colidx]}):
                    _res = _mod.fit(disp=False, maxiter=200)
            except ConvergenceWarning:
                try:
                    _res = _mod.fit(disp=False, maxiter=200)
                except ConvergenceWarning as e:
                    print(e)
            idx_end = _data.shape[0] - 1
            idx_start = np.where(_data.index == _data_valid.index[-1])[0]
            if (idx_end - idx_start) > 1 and (idx_end - idx_start) < self.nan_limit:
                fh_start = _data.index[idx_start + 1][0]
                fh_end = _data.index[idx_end]
                forecast = _res.get_forecast(int(idx_end - idx_start))
                _data_fill.loc[fh_start:fh_end] = forecast.predicted_mean.loc[
                    fh_start:fh_end
                ]
            if self.fill_remaining:
                _data_fill.fillna(_data_fill.median(), inplace=True)
            X_filled[:, colidx] = _data_fill.values
        np.save(self.filename, X_filled)
        return X_filled


def eqRate(cat, fixed_time=None, fixed_nevents=None, enddate=datetime.utcnow()):
    """
    Compute earthquake rate.

    :param cat: A catalogue of earthquakes as returned by
                :method:`whakaaribn.load_ruapehu_earthquakes`
    :type cat: :class:`pandas.DataFrame`
    :param fixed_time: If not 'None', compute the earthquake rate
                       based on a fixed-length time window given in
                       days.
    :type fixed_time: int
    :param fixed_nevents: If not None, compute the earthquake rate
                          based on a fixed number of events.
    :type fixed_nevents: int
    :param enddate: The latest date of the time-series.
                    Mainly needed for testing.
    :type enddate: :class:`datetime.datetime`

    """
    if fixed_time is not None and fixed_nevents is not None:
        raise ValueError("Please define either 'fixed_time' or 'fixed_nevents'")

    dates = cat["origintime"].values
    if fixed_time is not None:
        ds = pd.Series(np.ones(len(dates)), index=dates)
        ds = pd.concat([ds, pd.Series([np.nan], index=[enddate])])
        ds.sort_index(inplace=True)
        ds = ds.rolling("{:d}D".format(fixed_time)).count() / fixed_time
        ds.index -= pd.Timedelta("{:d}D".format(int(fixed_time / 2.0)))
        return pd.DataFrame({"obs": ds})
    elif fixed_nevents is not None:
        nevents = dates.shape[0]
        aBin = np.zeros(nevents - fixed_nevents, dtype="datetime64[ns]")
        aRate = np.zeros(nevents - fixed_nevents)
        iS = 0
        for s in np.arange(fixed_nevents, nevents):
            i1, i2 = s - fixed_nevents, s
            dt = (dates[i2] - dates[i1]).astype("timedelta64[s]")
            dt_days = dt.astype(float) / 86400.0
            aBin[iS] = dates[i1] + 0.5 * dt
            aRate[iS] = fixed_nevents / dt_days
            iS += 1
        return pd.DataFrame({"obs": aRate}, index=aBin)
    else:
        raise ValueError("Please define either 'fixed_time' or 'fixed_nevents'")


def reindex(df, dates, fill_method=None, ffill_interval=14):
    """
    Reindex and forward fill to generate a
    timeseries that can be used to set the evidence
    for a BN.

    :param df: Dataframe to reindex
    :type df: :class:`pandas.DataFrame`
    :param dates: new date index
    :type dates: :class:`pandas.DataTimeIndex`
    :param ffill_interval: Best-by interval for data
    :type ffill_interval: int
    """
    if fill_method is None:
        return df["obs"].resample("D").max().reindex(dates)
    elif fill_method == "ffill":
        return df["obs"].reindex(dates, method="ffill", limit=ffill_interval)
    elif fill_method == "interpolate":
        df_tmp = df["obs"].resample("D").max().reindex(dates)
        return df_tmp.interpolate(method="linear")
    else:
        msg = "'fill_method' has to be one of "
        msg += "[None, 'ffill', 'interpolate']"
        raise ValueError(msg)


def moving_average(X, window_size=30, axis=None, nan=True):
    """
    Compute the moving average of a time-series.
    >>> a = np.arange(18)
    >>> moving_average(a.reshape(6,3), n=3, axis=0)
    array([[nan, nan, nan],
            [nan, nan, nan],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.],
            [ 9., 10., 11.],
            [12., 13., 14.]])

    >>> moving_average(np.arange(18), n=3)
    array([nan, nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.,
            12., 13., 14., 15., 16.])
    """
    if axis == 1:
        X = X.T
    ret = np.cumsum(X, dtype=float, axis=axis)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    ret /= window_size
    if nan:
        ret[: window_size - 1] = np.nan
    if axis == 1:
        ret = ret.T
    return ret


def hex_to_rgb(value, alpha=1.0):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip("#")
    lv = len(value)
    rgb_list = [int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)]
    rgb_list.append(alpha)
    return tuple(rgb_list)


def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return "#%02x%02x%02x" % (red, green, blue)


def get_color(idx, alpha=1.0, style="seaborn-v0_8-paper"):
    """Return a color from the matplotlib color cycle by index.

    Parameters
    ----------
    idx : int
        Index of the color to return.
    alpha : float, optional
        Opacity of the color between 0 and 1, by default 1.
    style : str, optional
        matplotlib style to choose colors from, by default 'bmh'

    Returns
    -------
    _type_
        _description_
    """
    matplotlib.style.use(style)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors_rgb = []
    for c in colors:
        colors_rgb.append(hex_to_rgb(c, alpha=alpha))
    return f"rgba{colors_rgb[idx]}"
