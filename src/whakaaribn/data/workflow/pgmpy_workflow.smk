import pandas as pd
from datetime import datetime, timezone
from whakaaribn.grid_search import grid_search, get_best_estimator

STARTDATE=datetime(2009, 1, 1, tzinfo=timezone.utc)
ENDDATE=datetime(2026, 1, 1, tzinfo=timezone.utc)

include: "rules/data.smk"

rule all:
    input:
        "plots/eruption_forecasts_whakaari.png",
        "plots/model_objective_function.png",
        "plots/forecast_sensitivity.png",
        "plots/forecast_uncertainty.png",
        "plots/forecast_ensemble.png",
        "plots/whakaari_data_plot.png",
        "plots/whakaari_island_and_graphs.png",
        "plots/eruption_forecasts_whakaari_hindcast.png"

rule grid_search:
    input:
        "data/whakaari_data_with_groups.csv"
    output:
        "results/grid_search_results.csv"
    run:
        data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        params_gcv = {2: {'discretize__bins': [[0, 5, 100], [0, 50, 100], [0, 95, 100]],
                          'clf__nstates': [2, 2, 2]},
                      3: {'discretize__bins': [[0, 5, 95, 100], [0, 33, 66, 100], [0, 25, 75, 100]],
                          'clf__nstates': [3, 3, 3]},
                      4: {'discretize__bins': [[0, 25, 50, 75, 100], [0, 5, 50, 95, 100], [0, 10, 50, 90, 100], [0, 20, 50, 80, 100]],
                          'clf__nstates': [4, 4, 4, 4]},
                      5: {'discretize__bins': [[0, 20, 40, 60, 80, 100], [0, 5, 20, 80, 95, 100], [0, 5, 25, 75, 95, 100]],
                          'clf__nstates': [5, 5, 5]}}  
        grid_search(data, params_gcv, fout=output[0], recompute=True, njobs=10)

rule forecast:
    input:
        "data/whakaari_data_with_groups.csv",
        "results/grid_search_results.csv"
    output:
        "forecasts/whakaari_forecasts_{label}.nc",
        "forecasts/whakaari_forecasts_{label}_hindcast.nc"
    run:
        from whakaaribn.forecast import forecast
        data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        bins, pew = get_best_estimator(input[1])
        forecast_params = {"all_data": [],
                        "seismic": ['SO2', 'H2S', 'CO2'],
                        "gas": ['RSAM', 'Eqr']}
        xds_best = forecast(data, bins=bins, pew=pew, smoothing=30,
                            exclude_from_test=forecast_params[wildcards.label])
        xds_best.to_netcdf(output[0])
        data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        xds_best_hindcast = forecast(data, bins=bins, pew=pew, smoothing=30,
                            exclude_from_test=forecast_params[wildcards.label],
                            hindcast=True)
        xds_best_hindcast.to_netcdf(output[1])


rule validation_plot:
    input:
        forecast="forecasts/whakaari_forecasts_all_data.nc",
        data="data/whakaari_data_with_groups.csv"
    output:
        "plots/whakaari_validation_plot.png"
    log:
        notebook="logs/notebooks/validation_plots.ipynb"
    notebook:
        "notebooks/validation_plots.py.ipynb"

rule sensitivity:
    input:
        "data/whakaari_data_with_groups.csv",
        "results/grid_search_results.csv"
    output:
        "forecasts/whakaari_sensitivity.nc"
    run:
        from whakaaribn.forecast import sensitivity_analysis
        bins, pew = get_best_estimator(input[1])
        data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        xds_sensitivity = sensitivity_analysis(data, pew=pew, bins=bins)
        xds_sensitivity.to_netcdf(output[0])


rule uncertainty:
    input:
        "results/grid_search_results.csv",
        "data/whakaari_data_with_groups.csv"
    output:
        "forecasts/whakaari_uncertainty.nc"
    run:
        from whakaaribn.forecast import uncertainty_analysis
        from sklearn import set_config
        set_config(transform_output="pandas")
        data = pd.read_csv(input[1], parse_dates=True, index_col=0)
        grid_search_results = pd.read_csv(input[0], index_col=(0, 1, 2), converters={"params": eval})
        xds_uncertainty = uncertainty_analysis(data, grid_search_results)
        xds_uncertainty.to_netcdf(output[0])

rule forecast_plots:
    input:
        forecast_all_data="forecasts/whakaari_forecasts_all_data.nc",
        forecast_seismic="forecasts/whakaari_forecasts_seismic.nc",
        forecast_gas="forecasts/whakaari_forecasts_gas.nc",
        forecast_all_data_hindcast="forecasts/whakaari_forecasts_all_data_hindcast.nc",
        forecast_seismic_hindcast="forecasts/whakaari_forecasts_seismic_hindcast.nc",
        forecast_gas_hindcast="forecasts/whakaari_forecasts_gas_hindcast.nc",
        data="data/whakaari_data_with_groups.csv",
    output:
        eruption_forecasts="plots/eruption_forecasts_whakaari.png",
        eruption_forecasts_hindcast="plots/eruption_forecasts_whakaari_hindcast.png"
    log:
        # optional path to the processed notebook
        notebook="logs/notebooks/forecast_plots.ipynb"
    notebook:
        "notebooks/forecast_plots.py.ipynb"

rule objective_function_plot:
    input:
        grid_search_results="results/grid_search_results.csv"
    output:
        model_objective_function="plots/model_objective_function.png",
    log:
        notebook="logs/notebooks/objective_function_plot.ipynb"
    notebook:
        "notebooks/objective_function_plot.py.ipynb"

rule sensitivity_plot:
    input:
        forecast_all_data="forecasts/whakaari_forecasts_all_data.nc",
        forecast_sensitivity="forecasts/whakaari_sensitivity.nc",
        data="data/whakaari_data_with_groups.csv"
    output:
        forecast_sensitivity_plot="plots/forecast_sensitivity.png",
    log:
        notebook="logs/notebooks/sensitivity_plot.ipynb"
    notebook:
        "notebooks/sensitivity_plot.py.ipynb"   

rule uncertainty_plot:
    input:
        forecast_all_data="forecasts/whakaari_forecasts_all_data.nc",
        forecast_uncertainty="forecasts/whakaari_uncertainty.nc",
        data="data/whakaari_data_with_groups.csv"
    output:
        forecast_uncertainty_plot="plots/forecast_uncertainty.png",
    log:
        notebook="logs/notebooks/uncertainty_plot.ipynb"
    notebook:
        "notebooks/uncertainty_plot.py.ipynb"   

rule ensemble_plot:
    input:
        forecast_all_data="forecasts/whakaari_forecasts_all_data.nc",
        forecast_uncertainty="forecasts/whakaari_uncertainty.nc",
        data="data/whakaari_data_with_groups.csv"
    output:
        forecast_ensemble_plot="plots/forecast_ensemble.png",
    log:
        notebook="logs/notebooks/ensemble_plot.ipynb"
    notebook:
        "notebooks/ensemble_plot.py.ipynb"