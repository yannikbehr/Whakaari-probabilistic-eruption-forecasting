rule download:
    output:
        "data/whakaari_data.csv"
    run:
        from aitana import whakaari
        start_date=STARTDATE
        end_date=ENDDATE
        data = whakaari.load_all(
            fill_method=None,
            start_date=start_date,
            end_date=end_date,
            ignore_data=("LP", "VLP"),
            fuse_so2=False
        )
        data.to_csv("data/whakaari_data.csv")

rule assign_groups:
    input:
        "data/whakaari_data.csv"
    output:
        "data/whakaari_data_with_groups.csv"
    run:
        from whakaaribn.util import assign_group_labels
        eruptions = whakaari.eruptions(2, "0D", end_date=ENDDATE)
        data = pd.read_csv(input[0], parse_dates=True, index_col=0)
        data_with_groups = assign_group_labels(data, eruptions,
        startdate=STARTDATE, enddate=ENDDATE,
        ndays=30, min_interval=360)
        data_with_groups.to_csv(output[0])

