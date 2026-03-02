from .util import (
    Bin,
    BinData,
    Discretizer,
    ForecastImputer,
    ForwardImputer,
    SequentialGroupSplit,
    assign_group_labels,
    bin_data,
    convert_probability,
    eqRate,
    get_color,
    gradient,
    hex_to_rgb,
    moving_average,
    pre_eruption_window,
    reindex,
    split_by_group,
)
from whakaaribn.model import WhakaariModel
import importlib
from os import PathLike
from typing import Optional


def get_data(filename: Optional[PathLike] = None) -> str:
    """Return path to whakaaribn package.

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
