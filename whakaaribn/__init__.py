import pkg_resources


def get_data(filename):
    return pkg_resources.resource_filename(__name__, filename)


from .bayesnet import BayesNet
from .data import (
    load_all_whakaari_data,
    load_whakaari_catalogue,
    load_whakaari_gas,
    load_whakaari_lp,
    load_whakaari_rsam,
    load_whakaari_so2,
    load_whakaari_vlp,
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
