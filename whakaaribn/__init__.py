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
    SequentialGroupSplit,
    assign_group_labels,
    split_by_group,
    pre_eruption_window
)


from whakaaribn.model import WhakaariModel