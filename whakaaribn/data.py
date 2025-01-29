from datetime import date, datetime, timedelta
from io import StringIO
import os

import numpy as np
import pandas as pd
import requests
from cachier import cachier

from whakaaribn import get_data
from whakaaribn.util import eqRate, gradient, reindex
from whakaaribn.assimilate import SO2FusionModel


def load_whakaari_rsam(
    startdate=datetime(2007, 1, 1, 0, 0, 0),
    enddate=datetime.utcnow(),
):
    """
    Load RSAM values NZ.WIZ.10.HHZ. This will download the data from Zenodo
    if it is not already available locally.

    Parameters:
    -----------
        :param startdate: The earliest date of the time-series.
        :type startdate: :class:`datetime.datetime`
        :param enddate: The latest date of the time-series.
                        Mainly needed for testing.
        :type enddate: :class:`datetime.datetime`

    Returns:
    --------
        :param df: Dataframe with RSAM values.
        :type df: :class:`pandas.DataFrame`
    """
    rsam_fn = get_data('data/RSAM_NZ.WIZ.10.HHZ.csv')
    if not os.path.isfile(rsam_fn):
        # download the data from zenodo
        print('Downloading RSAM data from Zenodo')
        url = "https://zenodo.org/records/14759090/files/RSAM_NZ.WIZ.10.HHZ.csv"
        response = requests.get(url)
        if response.status_code == 200:
            with open(rsam_fn, "wb") as f:
                f.write(response.content)
            print('Successfully downloaded RSAM data from Zenodo')
        else:
            print('Failed to download RSAM data from Zenodo')

    df = pd.read_csv(rsam_fn, skiprows=1, parse_dates=True, names=['dt', 'obs'], index_col=0,
                     date_format='ISO8601')
    df = df.tz_localize(None)
    if enddate is not None:
        df = df[df.index <= enddate]
    if startdate is not None:
        df = df[df.index >= startdate]
    return df


def fits_request(name: str, station: str, method: str):
    """
    Make a request to the FITS API.

    Parameters:
    -----------
        :param name: The name of the data type.
        :type name: str
        :param station: The station or sampling location.
        :type station str
        :param method: The sampling method.
        :type method: str
    Returns:
    --------
        :param df: Dataframe with the requested data.
        :type df: :class:`pandas.DataFrame`
    """
    #     # airborne cospec
    # url = fitsurl + '/observation?siteID=RU000&typeID=SO2-flux-a&methodID=cosp'
    # gasdata = pd.read_csv(url, names=names, skiprows=1, parse_dates=True, index_col='dt')

    # url = fitsurl + '/observation?siteID=RU000&typeID=SO2-flux-a&methodID=cont'
    # SO2_cont = pd.read_csv(url, names=names, skiprows=1, parse_dates=True, index_col='dt')
    names = ["dt", "obs", "err"]
    fitsurl = "https://fits.geonet.org.nz/observation"
    url = fitsurl + "?siteID={}&typeID={}&methodID={}"
    url = url.format(station, name, method)
    df = pd.read_csv(url, names=names, skiprows=1, parse_dates=True, index_col="dt")
    return df


def tilde_request(name: str, station: str, method: str, sensor: str = "MC01"):
    _tstart = str(date(2000, 1, 1))
    _tend = str((datetime.utcnow() - timedelta(days=29)).date())

    url1 = f"https://tilde.geonet.org.nz/v3/data/manualcollect/{station}/{name}/{sensor}/{method}/nil/latest/30d"
    url2 = f"https://tilde.geonet.org.nz/v3/data/manualcollect/{station}/{name}/{sensor}/{method}/nil/"
    url2 += f"{_tstart}/{_tend}"
    rval = requests.get(url1, headers={"Accept": "text/csv"})
    if rval.status_code != 200:
        msg = f"Download of {name} for {station} failed with status code {rval.status_code}"
        raise ValueError(msg)
    data = StringIO(rval.text)
    df_latest = pd.read_csv(
        data,
        index_col="timestamp",
        parse_dates=["timestamp"],
        usecols=["timestamp", "value", "error"],
    )
    # Make sure the index is a datetime object
    df_latest.index = pd.to_datetime(df_latest.index, format="ISO8601")

    rval = requests.get(url2, headers={"Accept": "text/csv"})
    if rval.status_code != 200:
        msg = f"Download of {name} for {station} failed with status code {rval.status_code}"
        raise ValueError(msg)
    data = StringIO(rval.text)
    df_historic = pd.read_csv(
        data,
        index_col="timestamp",
        parse_dates=["timestamp"],
        usecols=["timestamp", "value", "error"],
    )
    df_historic.index = pd.to_datetime(df_historic.index, format="ISO8601")
    if len(df_latest) > 0:
        df = df_historic.combine_first(df_latest)
    else:
        df = df_historic
    df.rename(columns={"value": "obs", "error": "err"}, inplace=True)
    df.index.name = "dt"
    return df


@cachier(stale_after=timedelta(days=1), cache_dir="~/.cache")
def load_whakaari_gas(
    species, enddate=datetime.utcnow(), startdate=None, method="cospec", station="WI000"
):
    """
    Query airborne gas flux data from FITS for Whakaari.

    Parameters:
    -----------
        :param species: Gas species to load. Can be either 'so2', 'co2', or 'h2s'.
        :type species: str
        :param startdate: The earliest date of the time-series.
        :type startdate: :class:`datetime.datetime`
        :param enddate: The latest date of the time-series.
        :type enddate: :class:`datetime.datetime`
        :param so2methodid: The methodID for SO2 data that is passed to FITS.
        :type so2methodid: str
        :param so2siteid: The siteID for SO2 data that is passed to FITS.
        :type so2siteid: str

    Returns:
    --------
        :param df: Dataframe with gas flux values and the associated error.
        :type df: :class:`pandas.DataFrame`
    """

    if species.lower() == "so2":
        try:
            gasdata = tilde_request("plume-SO2-gasflux", station, method)
            if enddate > datetime(2023, 11, 9) and method == "cospec":
                # no cospec data after this date
                method = "contouring"
                gasdata_ct = tilde_request(
                    "plume-SO2-gasflux", station, "contouring"
                ).loc["2023-11-10":]
                gasdata = gasdata.combine_first(gasdata_ct)
        except ValueError:
            gasdata = fits_request("SO2-flux-a", station, method)

    elif species.lower() == "co2":
        try:
            gasdata = tilde_request("plume-CO2-gasflux", station, method)
        except ValueError:
            gasdata = fits_request("CO2-flux-a", station, method)

    elif species.lower() == "h2s":
        try:
            gasdata = tilde_request("plume-H2S-gasflux", station, method)
        except ValueError:
            gasdata = fits_request("H2S-flux-a", station, method)
    else:
        raise ValueError("Gas species has to be one of 'so2', 'co2', 'h2s'")

    # convert to tons/day
    gasdata.obs = gasdata.obs * 86.4
    gasdata.err = gasdata.err * 86.4
    gasdata.index = pd.DatetimeIndex(gasdata.index)
    gasdata = gasdata.tz_localize(None)
    gasdata = gasdata[gasdata.index <= enddate]
    if startdate is not None:
        gasdata = gasdata[gasdata.index >= startdate]
    gasdata = gasdata.tz_localize(None)
    return gasdata


@cachier(stale_after=timedelta(days=1), cache_dir="~/.cache")
def _load_whakaari_so2(enddate=datetime.utcnow(), startdate=None):
    dataframes = {}
    for station in ["WI301", "WI302"]:
        for method in ["mdoas-ah", "mdoas-ch"]:
            key = "{:s}-{:s}".format(station, method)
            dataframes[key] = load_whakaari_gas(
                "so2",
                enddate=enddate,
                startdate=startdate,
                method=method,
                station=station,
            )
    for station in ["WI000"]:
        for method in ["cospec", "contouring"]:
            key = "{:s}-{:s}".format(station, method)
            dataframes[key] = load_whakaari_gas(
                "so2",
                enddate=enddate,
                startdate=startdate,
                method=method,
                station=station,
            )
    startdate = np.array([df.index.min() for df in list(dataframes.values())]).min()
    enddate = np.array([df.index.max() for df in list(dataframes.values())]).max()
    dates = pd.date_range(startdate, enddate, freq="1 D")
    gasdata = {}
    gasdata_err = {}
    interval = "1D"
    for key, df in dataframes.items():
        gasdata[key] = (
            df["obs"].groupby(pd.Grouper(freq=interval)).median().reindex(dates).values
        )
        # South Rim
        if key == "WI301-mdoas-ah":
            # gasdata_err[key] = df['obs'].groupby(pd.Grouper(freq=interval)).std().reindex(dates).values
            gasdata_err[key] = (
                np.ones(
                    df["obs"]
                    .groupby(pd.Grouper(freq=interval))
                    .std()
                    .reindex(dates)
                    .size
                )
                * 4.5
                * 86.4
            )
        elif key == "WI301-mdoas-ch":
            gasdata_err[key] = (
                np.ones(
                    df["obs"]
                    .groupby(pd.Grouper(freq=interval))
                    .std()
                    .reindex(dates)
                    .size
                )
                * 2.5
                * 86.4
            )
        # North Rim
        elif key == "WI302-mdoas-ah":
            gasdata_err[key] = (
                np.ones(
                    df["obs"]
                    .groupby(pd.Grouper(freq=interval))
                    .std()
                    .reindex(dates)
                    .size
                )
                * 2.5
                * 86.4
            )
        elif key == "WI302-mdoas-ch":
            gasdata_err[key] = (
                np.ones(
                    df["obs"]
                    .groupby(pd.Grouper(freq=interval))
                    .std()
                    .reindex(dates)
                    .size
                )
                * 1.5
                * 86.4
            )
        else:
            gasdata_err[key] = df["err"].reindex(dates).values

    gasdata = pd.DataFrame(gasdata, index=dates)
    gasdata_err = pd.DataFrame(gasdata_err, index=dates)

    # Drop north-east point data as it is reporting consistently too low values
    gsd = gasdata.drop("WI301-mdoas-ah", axis=1)
    gsd_err = gasdata_err.drop("WI301-mdoas-ah", axis=1)
    return (gsd, gsd_err)


def load_whakaari_so2(
    startdate=None, enddate=datetime.utcnow(), fuse=True, smooth=False,
    ignore_cache=False
):
    """
    Load and optionally fuse SO2 observations from Whakaari.

    Parameters:
    -----------
        :param startdate: The earliest date of the time-series.
        :type startdate: :class:`datetime.datetime`
        :param enddate: The latest date of the time-series.
                        Mainly needed for testing.
        :type enddate: :class:`datetime.datetime`
        :param fuse:  Use Kalman Filter to fuse sensor and airborne data.
        :type fuse: bool
        :type smooth: bool
        :param smooth: Use Kalman Smoother to smooth the data.
        :type ignore_cache: bool
        :param ignore_cache: Ignore the cache and force a new download.


    Returns:
    --------
        :param df: Dataframe with SO2 flux values and the associated error.
        :type df: :class:`pandas.DataFrame`
    """

    if not fuse:
        return load_whakaari_gas(
            "so2",
            enddate=enddate,
            startdate=startdate,
            method="cospec",
            station="WI000",
        )
    
    gsd, gsd_err = _load_whakaari_so2(
        enddate=enddate, startdate=startdate, ignore_cache=ignore_cache
    )

    # Sensor fusion using Kalman Filter/Smoother
    try:
        # Drop data that is consistently too low
        gsd = gsd.drop(columns=['WI302-mdoas-ch', 'WI302-mdoas-ah'])
    except KeyError:
        pass
    # define a covariance matrix for the observations that gives cospec values
    # a higher weight than mdoas values
    obs_cov = np.diag(np.array([200., 5., 20.])**2)
    model = SO2FusionModel(measurements=gsd, initial_state=np.array([0]), initial_cov=np.diag([[1]]),
                           obs_cov=obs_cov, k_states=1, k_posdef=1)
    # Set initial parameters for process noise
    initial_params = [0.5]
    # Fit the model
    result = model.fit(start_params=initial_params, disp=False)

    # Get the smoothed state estimates (filtered values)
    filtered_state = result.filtered_state[0]
    smoothed_state = result.smoothed_state[0]

    mean_trace = filtered_state
    error = np.sqrt(result.filtered_state_cov[0, 0])
    if smooth:
        mean_trace = smoothed_state
        error = np.sqrt(result.smoothed_state_cov[0, 0])
    so2df = pd.DataFrame({"obs": mean_trace, "err": error}, index=gsd.index)
    so2df.index.name = "dt"
    return so2df


@cachier(stale_after=timedelta(days=1), cache_dir="~/.cache")
def load_whakaari_earthquakes(
    startdate=datetime(2005, 1, 1), enddate=datetime.utcnow()
):
    startdate = startdate.strftime("%Y-%m-%dT%H:%M:%S.0Z")
    maxdepth = "30"
    whakaari_pt = "177.186833+-37.523118"
    url = "http://wfs.geonet.org.nz/geonet/ows?service=WFS&version=1.0.0"
    url += (
        "&request=GetFeature&typeName=geonet:quake_search_v1&outputFormat={oFormat:s}"
    )
    url += "&cql_filter=origintime>={origintime:s}"
    url += "+AND+DWITHIN(origin_geom,Point+({point:s}),{radius:d},meters)+AND+depth<{depth:s}"
    req = url.format(
        oFormat="csv",
        origintime=startdate,
        radius=20000,
        point=whakaari_pt,
        depth=maxdepth,
    )
    cat = pd.read_csv(req, parse_dates=["origintime"])
    cat.sort_values(["origintime"], ascending=True, inplace=True)
    cat = cat.reset_index()
    cat = cat[cat.origintime <= str(enddate)]
    cat = cat[cat.origintime >= str(startdate)]
    if cat.size == 0:
        msg = "There are no earthquakes available for"
        msg += "the selected date range ({}, {})."
        msg = msg.format(str(startdate), str(enddate))
        raise ValueError(msg)
    return cat


def load_whakaari_catalogue(eruption_scale, dec_interval, exclude=True):
    """
    This function loads the eruption catalogue for White Island and
    declusters it.

    :param eruption_scale: The minimum eruption scale (between 0 and 4)
                           to use. This is currently ignored as the
                           White Island catalogue does not contain
                           eruption scale.
    :type eruption_scale: int
    :param dec_interval: The declustering interval, which is the minimum
                         distance in time between any two eruptions.
    :type dec_interval: :class:`pandas.DateOffset` or str
    :param datadir: Path that contains the catalogue file in csv format.
    :type datadir: str
    :returns: Declustered eruption catalogue
    :rtype: :class:`pandas.DataFrame`
    """
    eruptions = pd.read_csv(
        get_data("data/WhiteIs_Eruption_Catalogue.csv"),
        index_col="Date",
        parse_dates=True,
        comment="#",
    )

    # Select eruptions of a particular scale or larger
    eruptions = eruptions[eruptions.Activity_Scale >= eruption_scale].copy()

    # duplicate time index as a data column
    eruptions.insert(1, "tvalue", eruptions.index)

    # calculate intereruption event time
    delta = (eruptions["tvalue"] - eruptions["tvalue"].shift()).fillna(
        pd.Timedelta(seconds=0)
    )
    eruptions.insert(1, "delta", delta)
    eruptions.iloc[0, 1] = pd.Timedelta(dec_interval)
    # Exclude certain date ranges from calculations to ensure a more
    # Poisson-like process by removing long term eruption periods.
    if exclude:
        period = eruptions.index > "2004-01-01 00:00:00"
        eruptions = eruptions[period]

    eruptions.index = pd.DatetimeIndex(eruptions.index)
    eruptions = eruptions.tz_localize(None)
    return eruptions[(eruptions.delta >= dec_interval)]


@cachier(stale_after=timedelta(days=7), cache_dir="~/.cache")
def load_all_whakaari_data(
    fill_method="interpolate",
    ffill_interval=14,
    startdate=datetime(2005, 1, 1),
    enddate=datetime.utcnow(),
    ignore_all_caches=False,
    ignore_data=(),
    fuse_so2=False,
):
    cols = {}
    cat = load_whakaari_earthquakes(
        startdate=startdate, enddate=enddate, ignore_cache=ignore_all_caches
    )
    dft = eqRate(cat, fixed_time=7).resample("D").mean().interpolate()
    new_dates = pd.date_range(dft.index[0], enddate, freq="D")
    if "Eqr" not in ignore_data:
        cols["Eqr"] = reindex(dft, new_dates, fill_method=fill_method)
    if "RSAM" not in ignore_data:
        rsam = load_whakaari_rsam(enddate=enddate)
        cols["RSAM"] = reindex(rsam, new_dates, fill_method=fill_method)
    if "CO2" not in ignore_data:
        co2 = load_whakaari_gas(
            "co2", enddate=enddate, method="contouring", ignore_cache=ignore_all_caches
        )
        cols["CO2"] = reindex(co2, new_dates, fill_method=fill_method)
    if "H2S" not in ignore_data:
        h2s = load_whakaari_gas(
            "h2s", enddate=enddate, method="contouring", ignore_cache=ignore_all_caches
        )
        cols["H2S"] = reindex(h2s, new_dates, fill_method=fill_method)
    if "SO2" not in ignore_data:
        smooth = False
        if fill_method == "interpolate":
            smooth = True
            fuse_so2 = True
        so2 = load_whakaari_so2(
            enddate=enddate, fuse=fuse_so2, smooth=smooth, ignore_cache=ignore_all_caches
        )
        cols["SO2"] = reindex(so2, new_dates, fill_method=fill_method)
    rdf = pd.DataFrame(cols, index=new_dates)
    rdf = rdf[rdf.index <= str(enddate)]
    rdf = rdf[rdf.index >= str(startdate)]
    rdf = rdf.tz_localize(None)
    return rdf
