import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from whakaaribn import get_data


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
    cat.magnitude.where(cat.magnitude > 0., other=0.1, inplace=True)
    cat.magnitude /=10
    px.set_mapbox_access_token(open(get_data("data/.mapbox_token")).read())
    fig = px.scatter_mapbox(cat, lat="latitude", lon="longitude",
                            color="depth", size="magnitude",
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            size_max=size_max, zoom=10)
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
        fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2],
                            shared_yaxes=True, horizontal_spacing=.01)
    if df2 is None:
        df2 = df.data.copy()
    fig.add_trace(go.Scatter(x=df2.index, y=df2['obs'], mode='lines',
                             marker=dict(color='rgb(51, 102, 255)'),
                             showlegend=False, connectgaps=False), row=row, col=1)
    for l in df.bins:
        fig.add_trace(go.Scatter(x=df.data.index, y=np.ones(df.data.shape[0])*l,
                                 mode='lines', showlegend=False,
                                 line=dict(color='black', dash='dash', width=.5)), row=row, col=1)
    bin_width = np.diff(df.bins)
    bin_centers = (np.array(df.bins[:-1]) + np.array(df.bins[1:]))/2.
    fig.add_trace(go.Bar(x=df.marginals(), y=bin_centers,
                         orientation='h', width=bin_width, showlegend=False,
                         marker=dict(color='rgb(51, 102, 255)',
                                     line=dict(color='rgb(0,0,0)',
                                               width=2))), row=row, col=2)
    fig.update_yaxes(title_text=ylabel, row=row, col=1)
    return fig


def forecast_timeseries(prob, fout=None, title='Eruption probability',
                        param='e', fig=None, yaxis='y', xaxis='x',
                        elicitation=None, color1='rgba(255,65,77, .6)',
                        color2='rgba(255,65,77, .3)',
                        legend_name='Bayesian Network'):
    
    _min = prob[param+'mean'].values - 2*prob[param+'std'].values
    _min = np.where(_min < 0., 0., _min)
    _max = prob[param+'mean'].values + 2*prob[param+'std'].values
    _control = prob[param+'control']
    if fig is None:
        fig = make_subplots()
    fig.add_trace(go.Scatter(x=prob.index, y=_control, mode='lines',
                             name=legend_name, line_color=color1, line_dash='dash',
                             xaxis=xaxis, yaxis=yaxis))
    fig.add_trace(go.Scatter(x=prob.index, y=_min, mode='lines', marker=dict(color="#444"),
                             line=dict(width=0), showlegend=False,
                             xaxis=xaxis, yaxis=yaxis))
    fig.add_trace(go.Scatter(x=prob.index, y=_max, mode='lines', marker=dict(color="#444"),
                             line=dict(width=0), showlegend=False, fillcolor=color2,
                             fill='tonexty', xaxis=xaxis, yaxis=yaxis))
    
    if elicitation is not None:
        elicitation = elicitation.reindex(prob.index, method='ffill')
        fig.add_trace(go.Scatter(x=elicitation.index, y=elicitation['Best guess median'], mode='lines',
                                 name='Elicitation (best guess)', line_color='black', xaxis=xaxis,
                                 yaxis=yaxis))
        fig.add_trace(go.Scatter(x=elicitation.index, y=elicitation['"84th percentile"'], mode='lines',
                                 name='Elicitation (84th percentile)',
                                 line=dict(color='black', width=2, dash='dash'),
                                 xaxis=xaxis, yaxis=yaxis))

    fig.update_yaxes(title_text=title, secondary_y=False)
    if fout is not None:
        if fout.endswith('.html'):
            fig.write_html(fout)
        else:
            fig.write_image(fout)
    return fig


data_trans = {'TemperatureBin': {'name': 'Temperature', 'unit': u'[\N{DEGREE SIGN}C]'},
              'GradientBin': {'name': 'Temperature Gradient', 'unit': u'[\N{DEGREE SIGN}C/day]'},
              'RSAM': {'name': 'RSAM', 'unit': '[nm/s]'},
              'RSAM100': {'name': 'RSAM last 100 days', 'unit': '[nm/s]'},
              'CO2': {'name': u'Airborne CO\u2082', 'unit': '[t/day]'},
              'SO2': {'name': u'Airborne SO\u2082', 'unit': '[t/day]'},
              'H2S': {'name': u'Airborne H\u2082S', 'unit': '[t/day]'},
              'CO2_SO2': {'name': u'Airborne \u2202C/S ratio', 'unit': ''},
              'Mg': {'name': u'\u2202Mg\u00B2\u207A', 'unit': '[mg/l/day]'},
              'SO4': {'name': u'\u2202SO\u2084\u00B2\u207B', 'unit': '[mg/l/day]'},
              'Mg_ClBin': {'name': u'\u2202Mg\u00B2\u207A/Cl\u207B', 'unit': ''},
              'Mg_Na': {'name': u'\u2202Mg\u00B2\u207A/Na\u207A', 'unit': ''},
              'Mg_K': {'name': u'\u2202Mg\u00B2\u207A/K\u207A', 'unit': ''},
              'Mg_Al': {'name': u'\u2202Mg\u00B2\u207A/Al\u207B', 'unit': ''},
              'Eqr_outer': {'name': 'Earthquake rate (outer area)', 'unit': '[1/day]'},
              'Eqr_inner': {'name': 'Earthquake rate (inner area)', 'unit': '[1/day]'},
              'Eqr': {'name': 'Earthquake rate', 'unit': '[1/day]'},
              'LP': {'name': 'Number of LPs per day', 'unit': '[1/day]'},
              'VLP': {'name': 'Number of VLPs per day', 'unit': '[1/day]'},
              }

              
def composite_plot(prob, data, cols=['RSAM', 'CO2', 'TemperatureBin'], elicitation=None,
                   fout=None):
    colors = ['rgba(123,204,196,1)',
              'rgba(50,136,189,1)',
              'rgba(253,174,97,1)']
    if prob is None:
        prob = pd.DataFrame({'emean': np.zeros(data.shape[0]),
                             'estd': np.zeros(data.shape[0]),
                             'econtrol': np.zeros(data.shape[0]),
                             'smean': np.zeros(data.shape[0]),
                             'sstd': np.zeros(data.shape[0]),
                             'scontrol': np.zeros(data.shape[0]),
                             'mmean': np.zeros(data.shape[0]),
                             'mstd': np.zeros(data.shape[0]),
                             'mcontrol': np.zeros(data.shape[0])},
                            index=data.index)
    fig = forecast_timeseries(prob, elicitation=elicitation, fout=None)
    yaxes = []
    yaxis_position = [0.83, 0.9, 0.96]
    for i in range(3):
        yaxes.append(dict(domain=[0, .8], title='', titlefont=dict(color=colors[i]),
                          tickfont=dict(color=colors[i]), anchor='free', overlaying='y',
                          side='right', position=yaxis_position[i], showgrid=False))
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines',
                                 line_color=colors[i], yaxis='y{:d}'.format(i+2),
                                 name=data_trans[col]['name']))
        title = u'{:s} {:s}'.format(data_trans[col]['name'], data_trans[col]['unit'])
        yaxes[i]['title'] = title      


    forecast_timeseries(prob, param='s', elicitation=None, yaxis='y5', fout=None,
                        fig=fig, legend_name='Hydrothermal Seal',
                        color1='rgba(51,160,44,.6)', color2='rgba(51,160,44,.3)')
    forecast_timeseries(prob, param='m', elicitation=None, yaxis='y5', fout=None,
                        fig=fig, legend_name='Magmatic Intrusion',
                        color1='rgba(153,0,0,.6)', color2='rgba(153,0,0,.3)')
    fig.update_layout(
        xaxis=dict(
             domain=[0., 0.83]
        ),
        yaxis=dict(
            domain=[0, .75]
        ),
        yaxis2=yaxes[0],
        yaxis3=yaxes[1],
        yaxis4=yaxes[2], 
        yaxis5=dict(
            domain=[0.8, 1]
        ),
        margin=dict(b=10, t=10, l=20),
        font_size=14,
        template='plotly_white')
    if fout is not None:
        if fout.endswith('.html'):
            fig.write_html(fout)
        else:
            fig.write_image(fout, scale=3)
    return fig