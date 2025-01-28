import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import warnings
from scipy.stats import gaussian_kde

app = dash.Dash(__name__)
warnings.filterwarnings("ignore")

app.layout = html.Div([
    html.H1("Interactive Epilepsy Seizure diary Data Dashboard", style={'text-align': 'left','fontSize': '40px'}),

    html.Div([
        # -------------------------
        # LEFT COLUMN
        # -------------------------
        html.Div([
            # Basic controls for data range and binning
            html.Label("Start Date:"),
            dcc.DatePickerSingle(
                id='start-date',
                date=(datetime.date.today() - datetime.timedelta(days=730))
            ),
            html.Label("     End Date:"),
            dcc.DatePickerSingle(
                id='end-date',
                date=datetime.date.today(),style={'marginBottom': '20px'}
            ),
        html.Div([
            html.Label("Seizures to Display:",style={'marginBottom': '20px','fontWeight': 'bold',  # makes the text bold
        'fontSize': '18px'})]),
            dcc.Dropdown(
                id='seizure-types',
                options=[
                    {'label': 'Standard nighttime Seizure', 'value': 'Standard nighttime Seizure'},
                    {'label': 'Seizure from Wakefulness', 'value': 'Seizure from Wakefulness'},
                    {'label': 'Have continuous focal SE', 'value': 'Have continuous focal SE'}
                ],
                value=['Standard nighttime Seizure', 'Seizure from Wakefulness'],
                multi=True,
                style={'marginBottom': '20px'}
        ),
            html.Label("Heatmap Binning (X-axis):",
    style={
        'fontWeight': 'bold',  # makes the text bold
        'fontSize': '16px'     # optionally increase font size, etc.
    }),
            dcc.RadioItems(
                id='binning',
                options=[
                    {'label': 'Weekly', 'value': 'Weekly'},
                    {'label': 'Monthly', 'value': 'Monthly'},
                    {'label': 'Daily', 'value': 'Daily'}
                ],
                value='Weekly',
                inline=True,
                style={'marginBottom': '20px'}
            ),
            html.Label("Show Meds:",
    style={
        'fontWeight': 'bold',  # makes the text bold
        'fontSize': '16px'     # optionally increase font size, etc.
    })
                        

                    ,
            dcc.RadioItems(
                id='show-meds',
                options=[
                    {'label': 'Yes', 'value': 'Yes'},
                    {'label': 'No', 'value': 'No'}
                ],
                value='No',
                inline=True,
                style={'marginBottom': '20px'}
            ),
            dcc.Checklist(
                id='needed-rescue',
                options=[{'label': 'Needed Rescue Meds', 'value': 'Yes'}],
                value=[]
            ),

            # Graphs for the left column
            dcc.Graph(
                id='bar-chart',
                style={'height': '180px', 'margin-top': '5px'}
            ),
            dcc.Graph(
                id='heatmap',
                style={'height': '600px', 'margin-top': '20px'},
                config={'displayModeBar': True}
            )
        ], style={'width': '50%', 'padding': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'}),

        # -------------------------
        # RIGHT COLUMN
        # -------------------------
        html.Div([
            html.H4(
                "Data Display from Heatmap selection: Use Comparison mode to compare time periods", 
                style={
                    'text-align': 'left',
                    'font-size': '22px',
                    'margin-bottom': '20px'
                }
            ),

            # Mode Toggle (Exploratory vs. Comparison), plus Reset
            html.Div([
                html.Label("Mode:", style={'font-size': '20px', 'margin-right': '10px','fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='mode-toggle',
                    options=[
                        {'label': 'Exploratory', 'value': 'exploratory'},
                        {'label': 'Comparison', 'value': 'comparison'}
                    ],
                    value='exploratory',
                    style={'font-size': '20px', 'display': 'inline-block'}
                ),
                html.Button(
                    "Reset Comparison", 
                    id='reset-button', 
                    n_clicks=0, 
                    style={
                        'fontSize': '20px', 
                        'margin-left': '20px', 
                        'padding': '6px 12px'
                    }
                )
            ], style={'margin-bottom': '20px', 'textAlign': 'left'}),

            # Analysis Range dropdown: how big a time window do we fetch upon a click
            html.Div([
                html.Label("Analysis Range:", style={'font-size': '16px', 'margin-right': '10px'}),
                dcc.Dropdown(
                    id='analysis-range',
                    options=[
                        {'label': 'Same as Heatmap Binning', 'value': 'same'},
                        {'label': 'Full Month', 'value': 'month'},
                        {'label': '2 Weeks', 'value': '2weeks'},
                    ],
                    value='month',
                    clearable=False,
                    style={'width': '220px'}
                ),
            ], style={'margin-bottom': '20px', 'textAlign': 'left'}),

            # The detail graphs
            dcc.Graph(id='daily-graph', style={'textAlign': 'left'}),
            dcc.Graph(id='hourly-graph', style={'textAlign': 'left'})
        ], style={
            'width': '50%', 
            'padding': '10px', 
            'textAlign': 'left'
        })
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'}),

    # dcc.Store for comparison data
    dcc.Store(id='compare-store', data={})
],
style={'fontFamily': 'Arial, sans-serif'}
)


# ------------------------------------------------------------------------------
# 1) Callback for Bar-Chart and Heatmap
#    We also read 'compare-store' to highlight selected periods with transparent shapes.
# ------------------------------------------------------------------------------
@app.callback(
    Output('bar-chart', 'figure'),
    Output('heatmap', 'figure'),
    Input('start-date', 'date'),
    Input('end-date', 'date'),
    Input('seizure-types', 'value'),
    Input('binning', 'value'),
    Input('show-meds', 'value'),
    Input('needed-rescue', 'value'),
    Input('compare-store', 'data')  # needed to add highlighting shapes
)
def update_bar_and_heatmap(
    start_date,
    end_date,
    seizure_types,
    binning,
    show_meds,
    needed_rescue,
    compare_data
):
    """Produces the left-column bar chart & heatmap. Also adds shape overlays
    to highlight the selected time ranges from the compare-store data."""
    # 1) Load & Filter
    df = pd.read_csv('When-Did-I.csv', index_col=2, parse_dates=True)
    df = df.loc[df['Note'] != "NHNN VT"]
    df = df[df['Action'].isin(seizure_types)]
    df['datetime'] = df.index
    df['hour'] = df['datetime'].dt.hour

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]

    # 2) Binning for the heatmap
    freq_map = {'Weekly': 'W', 'Monthly': 'M', 'Daily': 'D'}
    period_freq = freq_map[binning]
    all_periods = pd.period_range(start=start_date, end=end_date, freq=period_freq)
    df['period'] = df['datetime'].dt.to_period(freq=period_freq)

    grouped = df.groupby('period').apply(lambda x: np.histogram(x['hour'], bins=range(25))[0])
    grouped_df = pd.DataFrame(grouped.tolist(), index=grouped.index)
    complete_grouped_df = grouped_df.reindex(all_periods, fill_value=0)

    # 3) Bar Chart
    x_labels = complete_grouped_df.index.astype(str)
    y_counts = complete_grouped_df.sum(axis=1)

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=x_labels,
        y=y_counts,
        name='Seizure Counts'
    ))
    bar_fig.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis_title="Seizure Count",
        height=180,
        margin=dict(t=10, b=30)
    )

    # 4) Heatmap
    heatmap_data = complete_grouped_df.T[::-1]
    heatmap_fig = go.Figure()
    heatmap_fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=x_labels,
        y=list(range(24))[::-1],
        colorscale='Viridis'
    ))
    heatmap_fig.update_layout(
        xaxis=dict(title="Time Period",type='category'),
        yaxis_title="Hour of Day",
        yaxis=dict(tickmode='linear', tick0=0, dtick=1),
        height=600,
        margin=dict(t=20, b=20)
    )

    # 5) Highlight any selected time windows from compare_data
    shape_list = []
    first_data = compare_data.get('first_data', {})
    second_data = compare_data.get('second_data', {})

    if first_data:
        shape_1 = build_highlight_shape(
            first_data, 
            x_labels, 
            color='blue'
        )
        if shape_1:
            shape_list.append(shape_1)

    if second_data:
        shape_2 = build_highlight_shape(
            second_data, 
            x_labels, 
            color='red'
        )
        if shape_2:
            shape_list.append(shape_2)

    if shape_list:
        heatmap_fig.update_layout(shapes=shape_list)

    return bar_fig, heatmap_fig


# ------------------------------------------------------------------------------
# 2) Manage the compare-store (exploratory vs. comparison mode),
#    and use the "analysis-range" to expand the subset.
# ------------------------------------------------------------------------------
@app.callback(
    Output('compare-store', 'data'),
    Input('heatmap', 'clickData'),
    Input('mode-toggle', 'value'),
    Input('reset-button', 'n_clicks'),
    Input('analysis-range', 'value'),
    State('compare-store', 'data'),
    State('start-date', 'date'),
    State('end-date', 'date'),
    State('seizure-types', 'value'),
    State('binning', 'value'),
    State('show-meds', 'value'),
    State('needed-rescue', 'value')
)
def manage_store(
    clickData,
    mode_value,
    reset_n_clicks,
    analysis_range,
    store_data,
    start_date,
    end_date,
    seizure_types,
    binning,
    show_meds,
    needed_rescue
):
    """In Exploratory mode, each click overwrites the single dataset.
       In Comparison mode, the first click is 'first_data', second is 'second_data'.
       We also incorporate 'analysis-range' to expand beyond the single period if desired."""
    ctx = callback_context
    if not ctx.triggered:
        return store_data

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # (a) If user toggles mode or presses reset => clear store
    if trigger_id in ['mode-toggle', 'reset-button']:
        return {}

    # (b) Must be a heatmap click
    if not clickData or 'points' not in clickData:
        return store_data

    # 1) Load & Filter the Data
    df = pd.read_csv('When-Did-I.csv', index_col=2, parse_dates=True)
    df = df.loc[df['Note'] != "NHNN VT"]
    df = df[df['Action'].isin(seizure_types)]
    df['datetime'] = df.index
    df['hour'] = df['datetime'].dt.hour

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]

    freq_map = {'Weekly': 'W', 'Monthly': 'M', 'Daily': 'D'}
    period_freq = freq_map[binning]
    df['period'] = df['datetime'].dt.to_period(freq=period_freq)

    # 2) Identify the clicked period
    clicked_period_str = clickData['points'][0]['x']
    try:
        clicked_period = pd.Period(clicked_period_str, freq=period_freq)
    except:
        clicked_period = clicked_period_str

    # 3) Build an expanded subset if user chose "month", "2weeks", etc.
    df_sub = build_analysis_subset(df, clicked_period, analysis_range, period_freq)
    if df_sub.empty:
        return store_data

    # 4) Summaries for daily & hour samples
    daily_list, day_count = build_daily_indexed_counts(df_sub)
    hour_samples = df_sub['hour'].tolist()
    min_d = df_sub['datetime'].min()
    max_d = df_sub['datetime'].max()

    new_dataset = {
        # e.g. "2025-W02 (month)"
        'period_str': f"{clicked_period_str} ({analysis_range})",
        'daily_counts': daily_list,
        'day_count': day_count,
        'hour_samples': hour_samples,
        'min_date': min_d.isoformat(),
        'max_date': max_d.isoformat()
    }

    # 5) Store logic
    if mode_value == 'exploratory':
        # Overwrite the first_data, clear second
        return {
            'first_data': new_dataset,
            'second_data': {}
        }
    else:
        # comparison mode
        first_data = store_data.get('first_data', {})
        second_data = store_data.get('second_data', {})
        if not first_data:
            first_data = new_dataset
        else:
            second_data = new_dataset
        return {
            'first_data': first_data,
            'second_data': second_data
        }


# ------------------------------------------------------------------------------
# 3) Generate the Daily + Hourly detail plots
# ------------------------------------------------------------------------------
@app.callback(
    Output('daily-graph', 'figure'),
    Output('hourly-graph', 'figure'),
    Input('compare-store', 'data'),
    Input('mode-toggle', 'value')
)
def update_detail_figs(store_data, mode_value):
    """Daily bar chart + Hourly KDE chart, for either 1 or 2 datasets."""
    first_data = store_data.get('first_data', {})
    second_data = store_data.get('second_data', {})

    # ---------- DAILY ----------
    fig_daily = go.Figure(layout={'barmode': 'group'})
    if not first_data:
        fig_daily.update_layout(title="No data selected yet.   Click on heatmap:")
        fig_hourly = go.Figure()
        fig_hourly.update_layout(title="No data selected yet.  Click on heatmap:")
        return fig_daily, fig_hourly

    d1_counts = first_data['daily_counts']
    d1_len = first_data['day_count']
    x_1 = list(range(d1_len))

    if mode_value == 'exploratory' or not second_data:
        # Single dataset
        fig_daily.add_trace(go.Bar(
            x=x_1,
            y=d1_counts,
            marker_color='blue',
            name=f"Period {first_data['period_str']}"
        ))
        fig_daily.update_layout(
            title="Daily Seizure Counts",
            xaxis_title="Day Index",
            yaxis_title="Count"
        )
    else:
        # Two datasets
        d2_counts = second_data['daily_counts']
        d2_len = second_data['day_count']
        max_len = max(d1_len, d2_len)
        d1_padded = d1_counts + [0]*(max_len - d1_len)
        d2_padded = d2_counts + [0]*(max_len - d2_len)
        x_range = list(range(max_len))

        fig_daily.add_trace(go.Bar(
            x=x_range,
            y=d1_padded,
            marker_color='blue',
            name=f"Period {first_data['period_str']}"
        ))
        fig_daily.add_trace(go.Bar(
            x=x_range,
            y=d2_padded,
            marker_color='red',
            name=f"Period {second_data['period_str']}"
        ))
        fig_daily.update_layout(
            title="Daily Seizure Counts (Comparison)",
            xaxis_title="Day Index",
            yaxis_title="Count"
        )

    # ---------- HOURLY (KDE) ----------
    fig_hourly = go.Figure()
    if mode_value == 'exploratory' or not second_data:
        add_kde_trace(
            fig_hourly, 
            first_data['hour_samples'], 
            color='blue', 
            label=f"Period {first_data['period_str']}"
        )
        fig_hourly.update_layout(
            title="Hourly KDE Distribution",
            xaxis_title="Hour of Day (0-24)",
            yaxis_title="Density"
        )
    else:
        add_kde_trace(
            fig_hourly, 
            first_data['hour_samples'], 
            color='blue', 
            label=f"Period {first_data['period_str']}"
        )
        add_kde_trace(
            fig_hourly, 
            second_data['hour_samples'], 
            color='red', 
            label=f"Period {second_data['period_str']}"
        )
        fig_hourly.update_layout(
            title="Hourly KDE Distribution (Comparison)",
            xaxis_title="Hour of Day (0-24)",
            yaxis_title="Density"
        )

    return fig_daily, fig_hourly


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def build_analysis_subset(df, clicked_period, analysis_range, period_freq):
    """
    Expand the user's single clicked period to a larger date window if desired:
      - 'same' => exactly that period
      - 'month' => entire calendar month containing that period start
      - '2weeks' => from start_time for 14 days
    """
    # 1) Identify start/end of the clicked period
    if isinstance(clicked_period, pd.Period):
        start_t = clicked_period.start_time
        end_t = clicked_period.end_time
    else:
        return df.iloc[0:0]  # fallback, empty

    if analysis_range == 'same':
        # Filter to that exact period
        mask = (df['datetime'] >= start_t) & (df['datetime'] <= end_t)
        return df[mask]

    elif analysis_range == 'month':
        # entire calendar month containing start_t
        month_start = start_t.replace(day=1, hour=0, minute=0, second=0)
        if month_start.month == 12:
            next_month = month_start.replace(year=month_start.year+1, month=1)
        else:
            next_month = month_start.replace(month=month_start.month+1)
        month_end = next_month - pd.Timedelta(microseconds=1)
        mask = (df['datetime'] >= month_start) & (df['datetime'] <= month_end)
        return df[mask]

    elif analysis_range == '2weeks':
        real_end = start_t + pd.Timedelta(days=14) - pd.Timedelta(microseconds=1)
        mask = (df['datetime'] >= start_t) & (df['datetime'] <= real_end)
        return df[mask]

    else:
        # unknown option
        return df.iloc[0:0]

def build_daily_indexed_counts(df_sub):
    """Convert the data to day-indexed bar data."""
    if df_sub.empty:
        return [], 0
    smin = df_sub['datetime'].min().normalize()
    smax = df_sub['datetime'].max().normalize()
    all_days = pd.date_range(smin, smax, freq='D')
    df_sub['date_only'] = df_sub['datetime'].dt.date
    daily_counts = df_sub.groupby('date_only').size()
    daily_counts.index = pd.to_datetime(daily_counts.index)
    daily_counts = daily_counts.reindex(all_days, fill_value=0)
    return daily_counts.tolist(), len(all_days)

def add_kde_trace(fig, hour_samples, color='blue', label='Dataset'):
    """Plot a KDE curve of hour_samples with partial opacity, or markers if <2 samples."""
    if len(hour_samples) < 2:
        fig.add_trace(go.Scatter(
            x=hour_samples,
            y=[0.05]*len(hour_samples),
            mode='markers',
            marker_color=color,
            name=label + " (not enough for KDE)"
        ))
        return
    kde = gaussian_kde(hour_samples, bw_method='scott')
    x_grid = np.linspace(0, 24, 200)
    y_vals = kde(x_grid)
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=y_vals,
        mode='lines',
        fill='tozeroy',
        line=dict(color=color),
        opacity=0.4,
        name=label
    ))

def build_highlight_shape(dataset, x_labels, color='blue'):
    """
    Create a semi-transparent rectangle shape covering the columns (periods) that overlap
    [min_date, max_date]. We parse each x_label as a Period (W/M/D) and check overlap.
    """
    if 'min_date' not in dataset or 'max_date' not in dataset:
        return None
    min_d = pd.to_datetime(dataset['min_date'])
    max_d = pd.to_datetime(dataset['max_date'])
    if pd.isnull(min_d) or pd.isnull(max_d) or min_d > max_d:
        return None

    included_indices = []
    for i, lbl in enumerate(x_labels):
        # parse lbl => period
        p = None
        for attempt in ['D','W','M']:
            try:
                p = pd.Period(lbl, freq=attempt)
                break
            except:
                pass
        if not p:
            continue
        st = p.start_time
        en = p.end_time

        # overlap if st <= max_d and en >= min_d
        if (st <= max_d) and (en >= min_d):
            included_indices.append(i)

    if not included_indices:
        return None

    x0 = min(included_indices) - 0.5
    x1 = max(included_indices) + 0.5
    # Full vertical coverage: hour of day 0..24 => y0=-0.5, y1=23.5
    shape = dict(
        type="rect",
        xref="x",  
        yref="y",
        x0=x0,
        x1=x1,
        y0=-0.5,
        y1=23.5,
        fillcolor=color,
        opacity=0.3,
        layer="above",
        line_width=0
    )
    return shape

if __name__ == '__main__':
    app.run_server(debug=True)