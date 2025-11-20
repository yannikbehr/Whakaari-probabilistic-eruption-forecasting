import importlib
from os import PathLike
from typing import Optional


def get_data(filename: Optional[PathLike] = None) -> str:
    """Return path to zizou package.

    Parameters
    ----------
    filename : Pathlike, default None
        Append `filename` to returned path.

    Returns
    -------
    pkgdir_path

    """
    f = importlib.resources.files(__package__)
    return str(f) if filename is None else str(f / filename)


from .bayesnet import (
    BayesNet,
    circular_node_positions,
    stacked_node_positions,
    fully_connected,
    causal,
    create_network
)

from .util import (
    Bin,
    BinData,
    Discretizer,
    ForecastImputer,
    ForwardImputer,
    bin_data,
    convert_probability,
    eqRate,
    get_color,
    gradient,
    hash_dataframe,
    hex_to_rgb,
    moving_average,
    reindex,
)

from .forecast import (
    SequentialGroupSplit,
    WhakaariForecasts,
    WhakaariModel,
    get_group_labels,
    pre_eruption_window,
)

from .grid_search import (
    grid_search,
    get_roc_curve,
    make_strictly_increasing,
    evaluate_threshold,
    compute_rates
)

from .api import Forecast