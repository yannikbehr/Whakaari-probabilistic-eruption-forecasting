from datetime import datetime, timezone
import pandas as pd


STARTDATE=datetime(2009, 1, 1, tzinfo=timezone.utc)

N_JOBS = config.get("n_jobs", 1)

rule all:
    input:
        "results/whakaari_forecasts/best_model.nc",
        "results/whakaari_forecasts/median_ensemble.nc",
        "results/whakaari_forecasts/max_ensemble.nc",
        "results/whakaari_forecasts/min_ensemble.nc",
        "results/whakaari_forecasts/Eqr.nc",
        "results/whakaari_forecasts/RSAM.nc",
        "results/whakaari_forecasts/SO2.nc",
        "results/whakaari_forecasts/H2S.nc",
        "results/whakaari_forecasts/CO2.nc"


rule live_data:
    output:
        "data/whakaari_live_data.csv"
    run:
        from aitana.whakaari import load_all 
        data = load_all(fill_method=None, start_date=STARTDATE)
        data.to_csv(output[0])

rule assign_groups_live_data:
    input:
        "data/whakaari_live_data.csv"
    output:
        "data/whakaari_live_data_with_groups.csv"
    run:
        from aitana import whakaari
        from whakaaribn.util import assign_group_labels
        data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        eruptions = whakaari.eruptions(2, "0D", end_date=data.index[-1])
        data_with_groups = assign_group_labels(data, eruptions,
        startdate=data.index[0], enddate=data.index[-1],
        ndays=30, min_interval=360)
        data_with_groups.to_csv(output[0])

rule grid_search:
    input:
        "data/whakaari_data_with_groups.csv"
    output:
        "results/grid_search_results.csv"
    run:
        from whakaaribn.grid_search import grid_search
        import numpy as np
        data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        params_gcv = [
            {"discretize__bins": [[0, 5, 100], [0, 50, 100], [0, 95, 100]],
             "clf__nstates": [2], "clf__pew": np.arange(10, 110, 10)},
            {"discretize__bins": [[0, 5, 95, 100], [0, 33, 66, 100], [0, 25, 75, 100]],
             "clf__nstates": [3], "clf__pew": np.arange(10, 110, 10)},
            {"discretize__bins": [[0, 25, 50, 75, 100], [0, 5, 50, 95, 100], [0, 10, 50, 90, 100], [0, 20, 50, 80, 100]],
             "clf__nstates": [4], "clf__pew": np.arange(10, 110, 10)},
            {"discretize__bins": [[0, 20, 40, 60, 80, 100], [0, 5, 20, 80, 95, 100], [0, 5, 25, 75, 95, 100]],
             "clf__nstates": [5], "clf__pew": np.arange(10, 110, 10)},
        ]
        search_results = grid_search(data, params_gcv, recompute=True, njobs=N_JOBS)
        search_results.to_csv(output[0], index=False)

rule forecast:
    input:
        "data/whakaari_live_data_with_groups.csv",
        "results/grid_search_results.csv"
    output:
        "forecasts/whakaari_live_forecasts.nc",
    run:
        from whakaaribn.forecast import forecast
        from whakaaribn.grid_search import get_best_estimator
        data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        bins, pew = get_best_estimator(input[1])
        xds_best = forecast(data, bins=bins, pew=pew, smoothing=30)
        xds_best.to_netcdf(output[0])

rule uncertainty:
    input:
        "results/grid_search_results.csv",
        "data/whakaari_live_data_with_groups.csv"
    output:
        "forecasts/whakaari_live_uncertainty.nc"
    run:
        from whakaaribn.forecast import uncertainty_analysis
        data = pd.read_csv(input[1], parse_dates=True, index_col=0)
        grid_search_results = pd.read_csv(input[0], index_col=(0, 1, 2), converters={"params": eval})
        xds_uncertainty = uncertainty_analysis(data, grid_search_results, n_jobs=N_JOBS)
        xds_uncertainty.to_netcdf(output[0])

rule save_results:
    input:
        forecast="forecasts/whakaari_live_forecasts.nc",
        uncertainty="forecasts/whakaari_live_uncertainty.nc",
    output:
        "results/whakaari_forecasts/best_model.nc",
        "results/whakaari_forecasts/median_ensemble.nc",
        "results/whakaari_forecasts/max_ensemble.nc",
        "results/whakaari_forecasts/min_ensemble.nc"
    params:
        outdir="results"
    run:
        import xarray as xr
        from tonik import Storage
        forecast_store = Storage('whakaari_forecasts', params.outdir)
        with xr.open_dataset(input.forecast) as ds:
            xds_best = ds.load()
        with xr.open_dataset(input.uncertainty) as ds:
            xds_ensemble = ds.load()
        datasets = {}
        datasets['best_model'] = (["datetime"], xds_best['probs'].data)
        median_ens = xds_ensemble.median('model_score').to_array().squeeze('variable')
        min_ens = xds_ensemble.chunk(dict(model_score=-1)).min('model_score').to_array().squeeze('variable')
        max_ens = xds_ensemble.chunk(dict(model_score=-1)).max('model_score').to_array().squeeze('variable')
        datasets['median_ensemble'] = (["datetime"], median_ens.data)
        datasets['max_ensemble'] = (["datetime"], max_ens.data)
        datasets['min_ensemble'] = (["datetime"], min_ens.data)
        xds = xr.Dataset(datasets, coords={"datetime": xds_best.datetime})
        forecast_store.save(xds)

rule save_data:
    input:
        "data/whakaari_live_data.csv"
    output:
        "results/whakaari_forecasts/Eqr.nc",
        "results/whakaari_forecasts/RSAM.nc",
        "results/whakaari_forecasts/SO2.nc",
        "results/whakaari_forecasts/H2S.nc",
        "results/whakaari_forecasts/CO2.nc"
    params:
        outdir="results"
    run:
        from tonik import Storage
        forecast_store = Storage('whakaari_forecasts', params.outdir)
        output_data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        output_data.index.name = 'datetime'
        output_data.index = output_data.index.tz_localize(None)
        forecast_store.save(output_data.to_xarray())
 