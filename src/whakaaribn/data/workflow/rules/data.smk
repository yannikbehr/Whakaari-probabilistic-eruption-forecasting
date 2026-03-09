start_date=STARTDATE
end_date=ENDDATE
 
rule download:
    output:
        "data/whakaari_data.csv"
    run:
        import pandas as pd
        from whakaaribn import get_data
        data = pd.read_csv(get_data("data/whakaaribn_data_release_v2.2.csv"), parse_dates=True, index_col=0)
        data.index = data.index.tz_localize("UTC")
        data.to_csv("data/whakaari_data.csv")

rule assign_groups:
    input:
        "data/whakaari_data.csv"
    output:
        "data/whakaari_data_with_groups.csv"
    run:
        from aitana import whakaari
        from whakaaribn.util import assign_group_labels
        eruptions = whakaari.eruptions(2, "0D", end_date=ENDDATE)
        data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        data_with_groups = assign_group_labels(data, eruptions,
        startdate=data.index[0], enddate=data.index[-1],
        ndays=30, min_interval=360)
        data_with_groups.to_csv(output[0])

rule data_plot:
    input:
        "data/whakaari_data_with_groups.csv"
    output:
        "plots/whakaari_data_plot.png"
    log:
        notebook="logs/notebooks/data_plot.ipynb"
    notebook:
        "../notebooks/data_plot.py.ipynb"

rule overview_plot:
    output:
        "plots/whakaari_island_and_graphs.png"
    log:
        notebook="logs/notebooks/overview_plot.ipynb"
    notebook:
        "../notebooks/overview_plot.py.ipynb"
