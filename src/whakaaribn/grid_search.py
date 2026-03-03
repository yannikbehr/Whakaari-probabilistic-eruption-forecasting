import os
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import pandas as pd
from aitana import whakaari
from sklearn import set_config
from sklearn.metrics import auc, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from whakaaribn import (
    Discretizer,
    SequentialGroupSplit,
    pre_eruption_window,
    split_by_group,
)
from whakaaribn.model import WhakaariModel
from whakaaribn.smile_model import PYSMILE_AVAILABLE, WhakaariSmileModel

set_config(transform_output="pandas")

# Validation Functions


def get_evaluation_windows(starttime, endtime, pew):
    """
    Get the evaluation windows for the forecasted rates.

    Arguments:
    ----------
        starttime: pd.Timestamp
            The start time of the forecast.
        endtime: pd.Timestamp
            The end time of the forecast.
        pew: int
            The pre-eruption window size.
    Returns:
    --------
        positive_windows: list
            The pre-eruption windows.
        negative_windows: list
            Non pre-eruption windows. 
    """
    explosive_eruptions = whakaari.eruptions(2, '0D', end_date=endtime)
    explosive_eruptions = explosive_eruptions.loc[starttime:endtime]
    positive_windows = []
    negative_windows = []
    tstart = starttime
    for e in explosive_eruptions.iterrows():
        positive_windows.append((e[0] - pd.Timedelta(days=pew-1), e[0]))
        negative_windows.append((tstart, e[0] - pd.Timedelta(days=pew)))
        tstart = e[0] + pd.Timedelta(days=30)
    negative_windows.append((tstart, endtime))
    return positive_windows, negative_windows


def compute_rates(model: pd.DataFrame, pew: int, debug: bool = False):
    """
    Compute the forecasted rates for positive and negative windows.

    Arguments:
    ----------
        model: pandas.DataFrame
            The model probabilities.
        pew: int
            The pre-eruption window size.
        debug: bool, optional
            Whether to print debug information.
    Returns:
    --------
        dict: The forecasted rates for positive and negative windows as well as the overall rate.
    """
    positive_windows, negative_windows = get_evaluation_windows(
        model.index[0], model.index[-1], pew)
    positive_rates = []
    for win_start, win_end in positive_windows:
        tmp_model = model.loc[win_start:win_end]
        rates = -np.log(1-tmp_model)
        if debug:
            print('--->', win_start, win_end, 'Forecasted rate:', rates)
        positive_rates.append(rates)

    negative_rates = []
    for win_start, win_end in negative_windows:
        tmp_model = model.loc[win_start:win_end]
        rates = -np.log(1-tmp_model)
        if debug:
            print('--->', win_start, win_end, 'Forecasted rate:', rates)
        negative_rates.append(rates)
    positive_rates = np.concatenate(positive_rates)
    negative_rates = np.concatenate(negative_rates)

    yearly_scale = 365.25/pew
    pos_mean = np.mean(positive_rates) * yearly_scale
    pos_sem = np.std(positive_rates) / \
        np.sqrt(len(positive_rates)) * yearly_scale
    neg_mean = np.mean(negative_rates) * yearly_scale
    neg_sem = np.std(negative_rates) / \
        np.sqrt(len(negative_rates)) * yearly_scale
    overall_rate = np.mean(-np.log(1-model)) * yearly_scale
    overall_sem = np.std(-np.log(1-model)) / np.sqrt(len(model)) * yearly_scale

    return dict(positive_rates=(pos_mean, pos_sem), negative_rates=(neg_mean, neg_sem),
                overall_rate=(overall_rate, overall_sem))


def evaluate_threshold(thresh: float, model: pd.DataFrame,
                       explosive_eruptions: pd.DataFrame,
                       debug: bool = False, pew: int = 90,
                       return_windows: bool = False):
    """
    Evaluate the number of true positives, false positives, true negatives and false negatives for a given threshold.

    Arguments:
    ----------
        thresh: float
            The threshold value.
        model: pandas.DataFrame
            The forecasted probabilities.
        debug: bool, optional
            Whether to print debug information.
        pew: int, optional
            The pre-eruption window size.
        return_windows: bool, optional
            Whether to return the positive and negative windows.

    Returns:
    --------
        dict: The evaluation results.
        list: A list of dictionaries for the tp, fp, tn, fn windows.
    """
    try:
        model.index = model.index.tz_localize('UTC')
    except TypeError:
        pass

    starttime = max(pd.Timestamp(
        model.index[0]), pd.Timestamp('2013-01-01', tz='UTC'))
    endtime = pd.Timestamp(model.index[-1])
    model = model.loc[starttime:endtime]
    explosive_eruptions = explosive_eruptions.loc[starttime:endtime]
    # normalise to 0-1
    model = (model - model.min()) / (model.max() - model.min())
    dt = pd.to_datetime(model.index)
    bin_model = np.where(model >= thresh, 1, 0)
    if len(bin_model.shape) > 1:
        bin_model = bin_model[0]
    # assign data on eruption days to the value on the day before
    eidx = np.where(np.isin(model.index, explosive_eruptions.index))[0]
    bin_model[eidx] = bin_model[eidx - 1]
    negative_windows = []
    positive_windows = []
    first_day = 0
    alert = bin_model[0]
    for i, date in enumerate(dt):
        if bin_model[i] != alert or i == len(dt) - 1:
            last_day = i - 1
            if i == len(dt) - 1:
                last_day = i
            if alert > 0:
                positive_windows.append((first_day, last_day))
            else:
                negative_windows.append((first_day, last_day))
            first_day = i
            alert = bin_model[i]

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    windows = []
    if debug:
        print('Positive windows:')
    for win_start, win_end in positive_windows:
        date_start = dt[win_start]
        date_end = dt[win_end]
        if debug:
            print('--->', date_start, date_end)
        explosion_in_window = False
        for ee in explosive_eruptions.iterrows():
            if date_start < ee[0] <= date_end:
                if pew is not None:
                    fp_ = tp_ = tn_ = fn_ = 0
                    pre_eruption_window = ee[0] - pd.Timedelta(days=pew)
                    tp_ += (date_end - ee[0]).days
                    tp_ += (ee[0] - max(pre_eruption_window, date_start)).days
                    pre_win = (date_start - pre_eruption_window).days
                    if pre_win < 0:
                        fp_ += abs(pre_win)
                    else:
                        fn_ += pre_win
                    false_positives += fp_
                    true_positives += tp_
                    false_negatives += fn_
                    # subtract from previous windows true_negatives
                    true_negatives -= fn_
                else:
                    true_positives += (win_end - win_start + 1)
                    windows.append(
                        dict(start=date_start, end=date_end, type='true_positive'))
                if debug:
                    print('Eruption in positive window: ', date_start, date_end)
                # stop if there is at least one eruption in the window
                explosion_in_window = True
                break
        if not explosion_in_window:
            false_positives += (win_end - win_start + 1)
            windows.append(
                dict(start=date_start, end=date_end, type='false_positive'))

    if debug:
        print('Negative windows:')
    for win_start, win_end in negative_windows:
        date_start = dt[win_start]
        date_end = dt[win_end]
        if debug:
            print('--->', date_start, date_end)
        explosion_in_window = False
        for ee in explosive_eruptions.iterrows():
            if date_start < ee[0] <= date_end:
                if pew is not None:
                    fp_ = tp_ = tn_ = fn_ = 0
                    tn_ = (date_end - ee[0]).days
                    pre_eruption_window = ee[0] - pd.Timedelta(days=pew)
                    fn_ = (ee[0] - max(pre_eruption_window, date_start)).days
                    pre_win = (date_start - pre_eruption_window).days
                    if pre_win < 0:
                        tn_ += abs(pre_win)
                    true_negatives += tn_
                    false_negatives += fn_
                else:
                    false_negatives += (win_end - win_start + 1)
                    # if window ends later than 09/01/2020 count the
                    # days between that date and the window end date as true negatives
                    # as the last explosive eruption was on 09/12/2019
                    if date_end > pd.Timestamp('2020-01-09', tz='UTC'):
                        diff = (
                            date_end - pd.Timestamp('2020-01-09', tz='UTC')).days
                        true_negatives += diff
                        false_negatives -= diff
                        windows.append(dict(
                            start=date_start, end=date_end - pd.Timedelta(days=diff), type='false_negative'))
                        windows.append(dict(
                            start=date_end - pd.Timedelta(days=diff-1), end=date_end, type='true_negative'))
                    else:
                        windows.append(
                            dict(start=date_start, end=date_end, type='false_negative'))
                if debug:
                    print('Eruption in negative window: ', date_start, date_end)
                # stop if there is at least one eruption in the window
                explosion_in_window = True
                break
        if not explosion_in_window:
            true_negatives += (win_end - win_start + 1)
            windows.append(
                dict(start=date_start, end=date_end, type='true_negative'))
    if return_windows:
        return dict(tp=true_positives, fp=false_positives, tn=true_negatives, fn=false_negatives), windows
    return dict(tp=true_positives, fp=false_positives, tn=true_negatives, fn=false_negatives)


def get_roc_curve(model: pd.DataFrame, thresholds: list, func: Callable, debug: bool = False):
    """
    Compute the ROC curve for a given model and thresholds.

    Arguments:
    ----------
        model: pandas.DataFrame
            The model probabilities.
        thresholds: list
            The thresholds to evaluate.
        func: function
            The evaluation function.
        debug: bool, optional
            Whether to print debug information.
    """
    tpr = np.empty(len(thresholds))*0.
    fpr = np.empty(len(thresholds))*0.
    precision = np.empty(len(thresholds))*0.

    for i, thresh in enumerate(thresholds):
        result = func(thresh, model)
        if debug:
            print(thresh, result)
        try:
            tpr_ = result['tp']/(result['tp'] + result['fn'])
            fpr_ = result['fp']/(result['fp'] + result['tn'])
            prec_ = result['tp']/(result['tp'] + result['fp'])
        except ZeroDivisionError:
            print("Threshold: ", thresh, "True positives: ",
                  result['tp'], "False negatives: ", result['fn'])
            print("Threshold: ", thresh, "True negatives: ",
                  result['tn'], "False positives: ", result['fp'])
            continue
        # start the evaluation from the first correct alerts
        if result['tp'] == 0:
            continue
        tpr[i] = tpr_
        fpr[i] = fpr_
        precision[i] = prec_
    # make sure that tpr and fpr end in 1
    # so that the AUC value is comparable
    tpr = np.r_[tpr, 1.]
    fpr = np.r_[fpr, 1.]
    return tpr, fpr, precision


def make_strictly_increasing(sequence: Sequence) -> Sequence:
    """
    Make a sequence strictly increasing. Some of the ROC curves computed with
    our own metric are not strictly increasing due to the way tps, fps, tns and fns
    are defined. This function makes sure that the sequence is strictly increasing so
    that we can caluculate the AUC value.

    Arguments:
    ----------
        sequence: Sequence
            The sequence to make strictly increasing.

    Returns:
    --------
        Sequence: The strictly increasing sequence.
    """
    # Make a copy to avoid modifying the original list
    result = sequence

    # Iterate through the sequence starting from the second element
    for i in range(1, len(result)):
        # If the current element is not greater than the previous one
        if result[i] <= result[i - 1]:
            # Increment the current element to be greater than the previous one
            result[i] = result[i - 1]

    return result


def ap(estimator, X, y, w=1):
    prob_e = estimator.predict_proba(X)
    weights = np.where(y == 1, w, 1)
    score = average_precision_score(y, prob_e[:, 1], sample_weight=weights,
                                    average='weighted')
    return score


def aic(estimator, X, y, dof=1):
    """
    Akaike Information Criterion
    """
    prob_e = estimator.predict_proba(X)
    score = -2*log_loss(y, prob_e[:, 1])
    score += 2*dof
    return score


def my_log_loss(estimator, X, y):
    prob_e = estimator.predict_proba(X)
    score = -log_loss(y, prob_e[:, 1])
    return score


def my_auc(estimator, X, y, w=1):
    prob_e = estimator.predict_proba(X)
    weights = np.where(y == 1, w, 1)
    score = roc_auc_score(y, prob_e[:, 1], sample_weight=weights,
                          average='weighted')
    return score


def my_roc_auc(estimator, X, y, eruptions, pew=90):
    prob_e = estimator.predict_proba(X)
    thresholds = np.linspace(0.01, 0.99, 100)[::-1]
    mdl = pd.Series(prob_e[:, 1], index=X.index)
    assert mdl.shape[0] > 0
    tpr_bn, fpr_bn, precision_bn = get_roc_curve(mdl, thresholds, partial(evaluate_threshold, pew=pew,
                                                                          explosive_eruptions=eruptions))
    score = auc(make_strictly_increasing(fpr_bn), tpr_bn)
    return score


def grid_search(data, params_gcv, fout=None, recompute=False, njobs=10, pews=np.arange(10, 110, 10),
                model_class: type[WhakaariModel | WhakaariSmileModel] = WhakaariModel):
    if model_class is WhakaariSmileModel and not PYSMILE_AVAILABLE:
        raise ImportError(
            "WhakaariSmileModel requires pysmile. "
            "Install it with: pip install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile"
        )
    if fout is not None and recompute is False:
        if os.path.exists(fout):
            print("Loading search results from ", fout)
            search_results = pd.read_csv(fout, index_col=(
                0, 1, 2), converters={'params': eval})
            return search_results

    pipe = Pipeline([('discretize', Discretizer()),
                     ('clf', model_class(smoothing=30, uniformize=True))])

    pipe.set_output(transform="pandas")
    cv = SequentialGroupSplit(data.group[data.group != 'e'])
    X_train, y_train, X_remain, y_remain = split_by_group(data, group='e')
    eruptions = whakaari.eruptions(2, '0D', end_date=data.index[-1])
    search_results = {}
    for pew in pews:
        print("Pre-eruption window = ", pew)
        for nstates, params in params_gcv.items():
            print("Number of states = ", nstates)
            dof = np.sum(2*nstates**np.arange(1, 6))
            _y_train = pre_eruption_window(y_train, pew)
            search = GridSearchCV(estimator=pipe, param_grid=[params],
                                  scoring={'average_precision': partial(ap, w=1),
                                           'aic': partial(aic, dof=dof),
                                           'log_loss': my_log_loss,
                                           'roc_auc': partial(my_auc, w=1),
                                           'mod_roc_auc': partial(my_roc_auc, pew=pew, eruptions=eruptions),
                                           'mod_roc_auc_no_pew': partial(my_roc_auc, pew=None, eruptions=eruptions)},
                                  cv=cv, n_jobs=njobs, verbose=0, refit=False)
            search.fit(X_train.ffill(), _y_train)
            sdf = pd.DataFrame(search.cv_results_)
            sdf = sdf.sort_values(by=['rank_test_mod_roc_auc_no_pew'])
            search_results[(pew, nstates)] = sdf

    search_results_combined = pd.concat(
        search_results, names=['pew', 'nstates'])
    search_results_combined.to_csv(fout)
    return search_results_combined


def get_best_estimator(grid_search_results: str) -> tuple:
    """Get the best estimator from the grid search results.

    Parameters
    ----------
    grid_search_results : str
        Path to the grid search results CSV file.
    Returns
    -------
    tuple
        The best estimator and the corresponding pre-eruption window size.
    """

    search_results = pd.read_csv(grid_search_results, index_col=(
        0, 1, 2), converters={'params': eval})
    best_pew, best_ns, best_params = search_results['mean_test_mod_roc_auc_no_pew'].idxmax(
    )
    best_estimator = search_results.loc[(
        best_pew, best_ns, best_params)].params
    return best_estimator['discretize__bins'], best_pew
