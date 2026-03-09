import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from aitana import whakaari
from plotly.subplots import make_subplots

from whakaaribn import get_color, get_data


def earthquake_map(cat, size_max=15):
    """
    Plot earthquake epicenters on an interactive map.
    :param cat: A pandas dataframe with the earthquake catalogue. It
                must have at least the columns 'latitude', 'longitude',
                'depth', and 'magnitude'.
    :type cat: :class:`pandas.DataFrame`
    :returns: 2D interactive map.
    :rtype: :class:`plotly.graph_objs.Figure`
    """

    # since magnitude is used for the size of the symbols
    # it can't be negative
    cat.magnitude.where(cat.magnitude > 0.0, other=0.1, inplace=True)
    cat.magnitude /= 10
    px.set_mapbox_access_token(open(get_data("data/.mapbox_token")).read())
    fig = px.scatter_mapbox(
        cat,
        lat="latitude",
        lon="longitude",
        color="depth",
        size="magnitude",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=size_max,
        zoom=10,
    )
    return fig


def plot_hist(df, ylabel, df2=None, fig=None, row=1):
    """
    Plot a binned timeseries.

    :param df: Data structure containing the original data and its bins.
    :type df: :class:`whakaaribn.BinData`
    :returns: Timeseries plot and data bins.
    :rtype: :class:`plotly.graph_objs.Figure`
    """
    if fig is None:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.8, 0.2],
            shared_yaxes=True,
            horizontal_spacing=0.01,
        )
    if df2 is None:
        df2 = df.data.copy()
    fig.add_trace(
        go.Scatter(
            x=df2.index,
            y=df2["obs"],
            mode="lines",
            marker=dict(color="rgb(51, 102, 255)"),
            showlegend=False,
            connectgaps=False,
        ),
        row=row,
        col=1,
    )
    for l in df.bins:
        fig.add_trace(
            go.Scatter(
                x=df.data.index,
                y=np.ones(df.data.shape[0]) * l,
                mode="lines",
                showlegend=False,
                line=dict(color="black", dash="dash", width=0.5),
            ),
            row=row,
            col=1,
        )
    bin_width = np.diff(df.bins)
    bin_centers = (np.array(df.bins[:-1]) + np.array(df.bins[1:])) / 2.0
    fig.add_trace(
        go.Bar(
            x=df.marginals(),
            y=bin_centers,
            orientation="h",
            width=bin_width,
            showlegend=False,
            marker=dict(
                color="rgb(51, 102, 255)", line=dict(color="rgb(0,0,0)", width=2)
            ),
        ),
        row=row,
        col=2,
    )
    fig.update_yaxes(title_text=ylabel, row=row, col=1)
    return fig


def forecast_timeseries(
    prob,
    fout=None,
    title="Eruption probability",
    param="e",
    fig=None,
    yaxis="y",
    xaxis="x",
    elicitation=None,
    color1="rgba(255,65,77, .6)",
    color2="rgba(255,65,77, .3)",
    legend_name="Bayesian Network",
):

    _min = prob[param + "mean"].values - 2 * prob[param + "std"].values
    _min = np.where(_min < 0.0, 0.0, _min)
    _max = prob[param + "mean"].values + 2 * prob[param + "std"].values
    _control = prob[param + "control"]
    if fig is None:
        fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=prob.index,
            y=_control,
            mode="lines",
            name=legend_name,
            line_color=color1,
            line_dash="dash",
            xaxis=xaxis,
            yaxis=yaxis,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prob.index,
            y=_min,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False,
            xaxis=xaxis,
            yaxis=yaxis,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prob.index,
            y=_max,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False,
            fillcolor=color2,
            fill="tonexty",
            xaxis=xaxis,
            yaxis=yaxis,
        )
    )

    if elicitation is not None:
        elicitation = elicitation.reindex(prob.index, method="ffill")
        fig.add_trace(
            go.Scatter(
                x=elicitation.index,
                y=elicitation["Best guess median"],
                mode="lines",
                name="Elicitation (best guess)",
                line_color="black",
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=elicitation.index,
                y=elicitation['"84th percentile"'],
                mode="lines",
                name="Elicitation (84th percentile)",
                line=dict(color="black", width=2, dash="dash"),
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

    fig.update_yaxes(title_text=title, secondary_y=False)
    if fout is not None:
        if fout.endswith(".html"):
            fig.write_html(fout)
        else:
            fig.write_image(fout)
    return fig


data_trans = {
    "TemperatureBin": {"name": "Temperature", "unit": "[\N{DEGREE SIGN}C]"},
    "GradientBin": {"name": "Temperature Gradient", "unit": "[\N{DEGREE SIGN}C/day]"},
    "RSAM": {"name": "RSAM", "unit": "[nm/s]"},
    "RSAM100": {"name": "RSAM last 100 days", "unit": "[nm/s]"},
    "CO2": {"name": "Airborne CO\u2082", "unit": "[t/day]"},
    "SO2": {"name": "Airborne SO\u2082", "unit": "[t/day]"},
    "H2S": {"name": "Airborne H\u2082S", "unit": "[t/day]"},
    "CO2_SO2": {"name": "Airborne \u2202C/S ratio", "unit": ""},
    "Mg": {"name": "\u2202Mg\u00b2\u207a", "unit": "[mg/l/day]"},
    "SO4": {"name": "\u2202SO\u2084\u00b2\u207b", "unit": "[mg/l/day]"},
    "Mg_ClBin": {"name": "\u2202Mg\u00b2\u207a/Cl\u207b", "unit": ""},
    "Mg_Na": {"name": "\u2202Mg\u00b2\u207a/Na\u207a", "unit": ""},
    "Mg_K": {"name": "\u2202Mg\u00b2\u207a/K\u207a", "unit": ""},
    "Mg_Al": {"name": "\u2202Mg\u00b2\u207a/Al\u207b", "unit": ""},
    "Eqr_outer": {"name": "Earthquake rate (outer area)", "unit": "[1/day]"},
    "Eqr_inner": {"name": "Earthquake rate (inner area)", "unit": "[1/day]"},
    "Eqr": {"name": "Earthquake rate", "unit": "[1/day]"},
    "LP": {"name": "Number of LPs per day", "unit": "[1/day]"},
    "VLP": {"name": "Number of VLPs per day", "unit": "[1/day]"},
}


def composite_plot(
    prob, data, cols=["RSAM", "CO2", "TemperatureBin"], elicitation=None, fout=None
):
    colors = ["rgba(123,204,196,1)", "rgba(50,136,189,1)",
              "rgba(253,174,97,1)"]
    if prob is None:
        prob = pd.DataFrame(
            {
                "emean": np.zeros(data.shape[0]),
                "estd": np.zeros(data.shape[0]),
                "econtrol": np.zeros(data.shape[0]),
                "smean": np.zeros(data.shape[0]),
                "sstd": np.zeros(data.shape[0]),
                "scontrol": np.zeros(data.shape[0]),
                "mmean": np.zeros(data.shape[0]),
                "mstd": np.zeros(data.shape[0]),
                "mcontrol": np.zeros(data.shape[0]),
            },
            index=data.index,
        )
    fig = forecast_timeseries(prob, elicitation=elicitation, fout=None)
    yaxes = []
    yaxis_position = [0.83, 0.9, 0.96]
    for i in range(3):
        yaxes.append(
            dict(
                domain=[0, 0.8],
                title="",
                titlefont=dict(color=colors[i]),
                tickfont=dict(color=colors[i]),
                anchor="free",
                overlaying="y",
                side="right",
                position=yaxis_position[i],
                showgrid=False,
            )
        )
    for i, col in enumerate(cols):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                mode="lines",
                line_color=colors[i],
                yaxis="y{:d}".format(i + 2),
                name=data_trans[col]["name"],
            )
        )
        title = "{:s} {:s}".format(
            data_trans[col]["name"], data_trans[col]["unit"])
        yaxes[i]["title"] = title

    forecast_timeseries(
        prob,
        param="s",
        elicitation=None,
        yaxis="y5",
        fout=None,
        fig=fig,
        legend_name="Hydrothermal Seal",
        color1="rgba(51,160,44,.6)",
        color2="rgba(51,160,44,.3)",
    )
    forecast_timeseries(
        prob,
        param="m",
        elicitation=None,
        yaxis="y5",
        fout=None,
        fig=fig,
        legend_name="Magmatic Intrusion",
        color1="rgba(153,0,0,.6)",
        color2="rgba(153,0,0,.3)",
    )
    fig.update_layout(
        xaxis=dict(domain=[0.0, 0.83]),
        yaxis=dict(domain=[0, 0.75]),
        yaxis2=yaxes[0],
        yaxis3=yaxes[1],
        yaxis4=yaxes[2],
        yaxis5=dict(domain=[0.8, 1]),
        margin=dict(b=10, t=10, l=20),
        font_size=14,
        template="plotly_white",
    )
    if fout is not None:
        if fout.endswith(".html"):
            fig.write_html(fout)
        else:
            fig.write_image(fout, scale=3)
    return fig


def trellis_plot(
    models: dict,
    data: pd.DataFrame,
    plot_uncertainty: bool = False,
    q_min: float = 0.15,
    q_max: float = 0.85,
):
    """Create a trellis plot for the given models and data.

    Parameters
    ----------
    models : dict
        A dictionary containing the models to plot. Each key should be a model name,
        and each value should be a dictionary with keys 'model', 'color', and optionally
        'colorscale' for ensemble models. The 'model' key should contain a xarray DataArray.
    data : pd.DataFrame
        A pandas DataFrame containing the data to plot.
        It should have a datetime index and a 'group' column indicating the group (b, c, d, e)
        for each time point.
    plot_uncertainty : bool, optional
        Whether to plot uncertainty for the models. Default is False.
    q_min : float, optional
        The minimum quantile to use for plotting uncertainty. Default is 0.15.
    q_max : float, optional
        The maximum quantile to use for plotting uncertainty. Default is 0.85.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly figure object containing the trellis plot.
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ],
    )

    showlegend = True
    for irow, group_name in enumerate(["b", "c", "d", "e"]):
        for name, model in models.items():
            if name in ["min", "max", "ensemble"]:
                continue
            time = pd.to_datetime(model["model"]["datetime"])[
                data.group == group_name]
            probs = model["model"].values[data.group == group_name]
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=probs,
                    mode="lines",
                    name=name,
                    line=dict(color=model["color"],
                              dash=model.get("dash", "solid")),
                    showlegend=showlegend,
                ),
                row=irow + 1,
                col=1,
            )
        if plot_uncertainty is not None:
            if plot_uncertainty == "quantile":
                probs_min = models["min"]["model"].values[data.group == group_name]
                probs_max = models["max"]["model"].values[data.group == group_name]
                fillcolor = models["min"]["color"]
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=probs_min,
                        mode="lines",
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False,
                    ),
                    row=irow + 1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=probs_max,
                        mode="lines",
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False,
                        fillcolor=fillcolor,
                        fill="tonexty",
                    ),
                    row=irow + 1,
                    col=1,
                )
            elif plot_uncertainty == "ensemble":
                for j in range(len(models["ensemble"]["model"].model_score)):
                    model_to_plot = (
                        models["ensemble"]["model"]
                        .isel(dict(model_score=j))
                        .values[data.group == group_name]
                    )
                    color = px.colors.sample_colorscale(
                        models["ensemble"]["colorscale"],
                        float(models["ensemble"]["model"].model_score[j]),
                        low=0.85,
                        high=9.5,
                    )[0]
                    fig.add_trace(
                        go.Scatter(
                            x=time,
                            y=model_to_plot,
                            mode="lines",
                            showlegend=False,
                            line=dict(color=color, width=0.1),
                            opacity=1.0,
                        ),
                        row=irow + 1,
                        col=1,
                    )

        # add eruption markers
        t1 = pd.Timestamp(time[0], tz="UTC")
        t2 = pd.Timestamp(time[-1], tz="UTC")
        dfe = whakaari.eruptions(end_date=data.index[-1]).loc[t1:t2]
        for i in range(len(dfe.index)):
            fig.add_trace(
                go.Scatter(
                    x=[dfe.index[i], dfe.index[i]],
                    y=[0.0, 1.0],
                    mode="lines",
                    line_width=0.8,
                    line_color="black",
                    name="Explosive Eruption",
                    showlegend=showlegend,
                ),
                secondary_y=True,
                row=irow + 1,
                col=1,
            )
            showlegend = False

        indices = np.where(data.group == group_name)[0]

    # Annotations
    x_annot = 0.08
    y_annot = 0.9
    fig.add_annotation(
        text="<b>(A) Group B</b>",
        xref="x domain",
        yref="y domain",
        x=x_annot,
        y=y_annot,
        showarrow=False,
    )
    fig.add_annotation(
        text="<b>(B) Group C</b>",
        xref="x3 domain",
        yref="y3 domain",
        x=x_annot,
        y=y_annot,
        showarrow=False,
    )
    fig.add_annotation(
        text="<b>(C) Group D</b>",
        xref="x3 domain",
        yref="y5 domain",
        x=x_annot,
        y=y_annot,
        showarrow=False,
    )
    fig.add_annotation(
        text="<b>(D) Group E</b>",
        xref="x3 domain",
        yref="y7 domain",
        x=x_annot,
        y=y_annot,
        showarrow=False,
    )

    fig.add_annotation(
        text="Dome extrusion", x="2012-11-24", y=1.2, showarrow=False, yref="y domain"
    )
    fig.add_vrect(
        x0="2012-11-22",
        x1="2012-12-10",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=1,
        col=1,
    )

    fig.add_annotation(
        text="Geysering", x="2013-02-15", y=1.2, showarrow=False, yref="y domain"
    )
    # Add shape regions
    fig.add_vrect(
        x0="2013-01-15",
        x1="2013-04-10",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=1,
        col=1,
    )

    fig.add_annotation(
        text="Minor steam and mud eruptions",
        x="2013-10-04",
        y=1.0,
        showarrow=True,
        yref="y domain",
        xanchor="right",
    )
    fig.add_annotation(x="2013-08-17", y=1.0, showarrow=True, yref="y domain")
    fig.add_vrect(
        x0="2013-08-15",
        x1="2013-08-18",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0="2013-10-01",
        x1="2013-10-08",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=1,
        col=1,
    )

    fig.add_annotation(
        text="Banded tremor",
        x="2015-10-13",
        y=1.2,
        showarrow=False,
        yref="y3 domain",
        xref="x2",
    )
    fig.add_vrect(
        x0="2015-10-13",
        x1="2015-10-20",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=2,
        col=1,
    )

    fig.add_annotation(
        text="Non-explosive ash venting",
        x="2016-09-13",
        y=1.2,
        showarrow=False,
        yref="y5 domain",
        xref="x3",
    )
    fig.add_vrect(
        x0="2016-09-13",
        x1="2016-09-18",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=3,
        col=1,
    )

    fig.add_annotation(
        text="Earthquake swarm",
        x="2019-06-15",
        y=1.2,
        showarrow=False,
        yref="y5 domain",
        xref="x3",
    )
    fig.add_vrect(
        x0="2019-04-23",
        x1="2019-07-01",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=3,
        col=1,
    )

    fig.add_annotation(
        text="Minor ash emissions",
        x="2019-12-24",
        y=1.2,
        showarrow=False,
        arrowcolor=get_color(7),
        yref="y5 domain",
        xref="x3",
    )
    fig.add_vrect(
        x0="2019-12-23",
        x1="2019-12-29",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=3,
        col=1,
    )

    fig.add_annotation(
        text="Lava extrusion",
        x="2020-01-15",
        y=1.2,
        showarrow=False,
        arrowcolor=get_color(7),
        yref="y7 domain",
        xref="x4",
    )
    fig.add_vrect(
        x0="2020-01-10",
        x1="2020-01-20",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=4,
        col=1,
    )

    fig.add_annotation(
        text="Minor ash emissions",
        x="2020-11-13",
        y=1.2,
        showarrow=False,
        arrowcolor=get_color(7),
        yref="y7 domain",
        xref="x4",
    )
    fig.add_vrect(
        x0="2020-11-13",
        x1="2020-12-01",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=4,
        col=1,
    )

    fig.add_annotation(
        text="Small steam explosions",
        x="2020-12-29",
        y=1.1,
        showarrow=False,
        xanchor="left",
        arrowcolor=get_color(7),
        yref="y7 domain",
        xref="x4",
    )
    fig.add_vrect(
        x0="2020-12-29",
        x1="2021-01-02",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=4,
        col=1,
    )

    fig.add_annotation(
        text="Minor ash emissions",
        x="2022-09-18",
        y=1.2,
        showarrow=False,
        yref="y7 domain",
        xref="x4",
    )
    fig.add_vrect(
        x0="2022-09-18",
        x1="2022-09-24",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=4,
        col=1,
    )

    fig.add_annotation(
        text="Small steam explosion",
        x="2024-05-24",
        y=1.2,
        showarrow=False,
        yref="y7 domain",
        xref="x4",
    )
    fig.add_vrect(
        x0="2024-05-24",
        x1="2024-05-31",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=4,
        col=1,
    )

    fig.add_annotation(
        text="Minor ash emissions",
        x="2024-07-24",
        y=1.1,
        showarrow=False,
        xanchor="left",
        yref="y7 domain",
        xref="x4",
    )
    fig.add_vrect(
        x0="2024-07-24",
        x1="2024-09-10",
        fillcolor=get_color(7),
        opacity=0.2,
        layer="below",
        line_width=0,
        row=4,
        col=1,
    )

    fig.update_layout(
        width=1200,
        height=1000,
        legend=dict(y=1.1, orientation="h", font=dict(size=15)),
    )
    fig.update_yaxes(range=[0, 1.2], row=5, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, secondary_y=True)
    fig.update_xaxes(range=["2012-09-01", "2024-06-19"], row=5, col=1)
    for _r in range(1, 5):
        fig.update_xaxes(tickfont=dict(size=15), row=_r, col=1)
        fig.update_yaxes(tickfont=dict(size=15), row=_r, col=1)
    return fig


def forecast_plot(
    frcst,
    frcst_min=None,
    frcst_max=None,
    fig=None,
    ploterr=True,
    log=False,
    eruptions=True,
    label="Forecast",
    showlegend=True,
    color_id=0,
    alpha=1.0,
    window=30,
    row=1,
    col=1,
):
    """
    Plot timeseries of forecast probabilities.
    Arguments:
    ----------
        frcst: pandas.DataFrame
            The forecast probabilities.
        fig: plotly Figure, optional
            The figure to add the plot to. If None, a new figure is created.
        ploterr: bool, optional
            Whether to plot error bounds of the forecast probabilities.
        log: bool, optional
            Whether to plot the y-axis on a log scale.
        eruptions: bool, optional
            Whether to plot the eruption times.
        label: str, optional
            The label of the forecast.
        color_id: int, optional
            The color of the forecast.
        window: int, optional
            The window size for the rolling mean.
        row: int, optional
            The row to add the plot to.
        col: int, optional
            The column to add the plot to.
    Returns:
    --------
        fig: plotly Figure
            The figure with the plot.

    """
    if fig is None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

    forecast = frcst
    if window is not None:
        forecast = frcst.rolling(dict(datetime=window)).mean()
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(forecast.datetime),
            y=forecast,
            mode="lines",
            line_color=get_color(color_id, alpha=alpha),
            name=label,
            showlegend=showlegend,
            legendgroup="group1",
        ),
        row=row,
        col=col,
    )
    if ploterr:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(frcst_min.datetime),
                y=frcst_min,
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(frcst_max.datetime),
                y=frcst_max,
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
                fillcolor=get_color(color_id, alpha=0.3),
                fill="tonexty",
            ),
            row=row,
            col=col,
        )
    if eruptions:
        eruptions = whakaari.eruptions(
            1, "0D", end_date=datetime.now(timezone.utc))
        dfe = eruptions.loc[
            pd.to_datetime(frcst.datetime[0].values, utc=True): pd.to_datetime(
                frcst.datetime[-1].values, utc=True
            )
        ]
        showlegend = showlegend
        for i in range(len(dfe.index)):
            fig.add_trace(
                go.Scatter(
                    x=[dfe.index[i], dfe.index[i]],
                    y=[0.0, 1.0],
                    mode="lines",
                    line_width=0.8,
                    line_color="black",
                    name="Observed eruption",
                    showlegend=showlegend,
                ),
                secondary_y=True,
                row=row,
                col=col,
            )
            showlegend = False

    if log:
        fig.update_yaxes(type="log", secondary_y=False)
    return fig
