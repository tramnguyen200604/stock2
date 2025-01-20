import os
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

folder_path = "Thailand_done"

file_path = 'DN.csv'

try:
    df_dn = pd.read_csv(file_path)
    dn_ticker_mapping = (
        df_dn.groupby("Full Name")["RIC"].apply(list).to_dict()
    )
    doanh_nghieps = sorted(dn_ticker_mapping.keys())
except FileNotFoundError:
    raise FileNotFoundError(f"File {file_path} not found.")


dn_dropdown = widgets.Dropdown(
    options=["Select a company"] + doanh_nghieps,
    description="Company:",
    disabled=False,
)

ticker_dropdown = widgets.Dropdown(
    options=["Select a company first"],
    description="Ticker:",
    disabled=True,
)

chart_type_dropdown = widgets.Dropdown(
    options=["Candlestick", "Line", "Classic Stock Chart"],
    description="Chart:",
    disabled=False,
)

display_type_dropdown = widgets.Dropdown(
    options=["Select Display Type", "Timeframe", "Year"],
    description="Display Type:",
    disabled=False,
)

timeframe_dropdown = widgets.Dropdown(
    options=["5Y (Monthly)", "1Y (Monthly)", "1 month (Weekly)"],
    description="Timeframe:",
    disabled=True,
)

year_dropdown = widgets.Dropdown(
    options=["Select a ticker first"],
    description="Year:",
    disabled=True,
)


page = widgets.Output()

def update_ticker_dropdown(change):
    selected_dn = change["new"]
    if selected_dn == "Select a company":
        ticker_dropdown.options = ["Select a company first"]
        ticker_dropdown.disabled = True
    else:
        tickers = dn_ticker_mapping.get(selected_dn, [])
        ticker_dropdown.options = tickers if tickers else ["No tickers found"]
        ticker_dropdown.disabled = False

dn_dropdown.observe(update_ticker_dropdown, names="value")

def get_years_for_ticker(ticker):
    file_path = os.path.join(folder_path, f"{ticker}.txt")
    if not os.path.exists(file_path):
        return []
    data = pd.read_csv(file_path, sep="\t", engine="python")
    data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
    years = sorted(data["Date"].dt.year.dropna().unique())
    return years

def update_year_dropdown(change):
    selected_ticker = change["new"]
    if selected_ticker != "No tickers found":
        years = get_years_for_ticker(selected_ticker)
        year_dropdown.options = years if years else ["No years found"]
        year_dropdown.disabled = False if years else True
    else:
        year_dropdown.options = ["Select a ticker first"]
        year_dropdown.disabled = True

ticker_dropdown.observe(update_year_dropdown, names="value")

def update_display_type(change):
    selected_display_type = change["new"]
    if selected_display_type == "Timeframe":
        timeframe_dropdown.disabled = False
        year_dropdown.disabled = True
    elif selected_display_type == "Year":
        timeframe_dropdown.disabled = True
        year_dropdown.disabled = False
    else:
        timeframe_dropdown.disabled = True
        year_dropdown.disabled = True

display_type_dropdown.observe(update_display_type, names="value")


def westerncandlestick(ax, quotes, width=0.2, colorup='k', colordown='r', linewidth=0.5):
    OFFSET = width / 2.0
    for q in quotes.values:
        t, open_, close, high, low = q[:5]
        t = mdates.date2num(t)
        color = colorup if close >= open_ else colordown
        ax.add_line(Line2D([t, t], [low, high], color=color, linewidth=linewidth))
        ax.add_line(Line2D([t - OFFSET, t], [open_, open_], color=color, linewidth=linewidth))
        ax.add_line(Line2D([t, t + OFFSET], [close, close], color=color, linewidth=linewidth))
    ax.autoscale_view()


def update_chart(change=None):
    with output:
        clear_output()
        selected_ticker = ticker_dropdown.value
        chart_type = chart_type_dropdown.value
        selected_display_type = display_type_dropdown.value
        selected_year = year_dropdown.value
        timeframe = timeframe_dropdown.value

        if selected_ticker and selected_ticker != "No tickers found":
            file_path = os.path.join(folder_path, f"{selected_ticker}.txt")
            try:
                data = pd.read_csv(file_path, sep="\t", engine="python")
                data["Date"] = pd.to_datetime(data["Date"], errors='coerce')

                if selected_display_type == "Timeframe":
                    end_date = data["Date"].max()
                    if timeframe == "5Y (Monthly)":
                        start_date = end_date - timedelta(days=5 * 365)
                    elif timeframe == "1Y (Monthly)":
                        start_date = end_date - timedelta(days=365)
                    elif timeframe == "1 month (Weekly)":
                        start_date = end_date - timedelta(days=30)
                    filtered_data = data[data["Date"] >= start_date]

                elif selected_display_type == "Year" and selected_year != "Select a ticker first":
                    filtered_data = data[data["Date"].dt.year == int(selected_year)]
                else:
                    filtered_data = pd.DataFrame()

                if filtered_data.empty:
                    print("No data available for the selected filters.")
                    return

                if chart_type == "Candlestick":
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=filtered_data["Date"],
                        open=filtered_data["Price Open"],
                        high=filtered_data["Price High"],
                        low=filtered_data["Price Low"],
                        close=filtered_data["Price Close"],
                        hovertext=filtered_data[["Price Open", "Price Close", "Price High", "Price Low", "Volume"]].apply(
                                lambda row: f"Open: {row['Price Open']}, Close: {row['Price Close']}, High: {row['Price High']}, Low: {row['Price Low']}, Volume: {row['Volume']}", axis=1
                            ),
                            hoverinfo="text",

                    ))
                    fig.show()
                elif chart_type == "Line":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=filtered_data["Date"],
                        y=filtered_data["Price Close"],
                        mode="lines",
                        line=dict(color="blue"),
                        hovertext=filtered_data[["Price Open", "Price Close", "Price High", "Price Low", "Volume"]].apply(
                                lambda row: f"Open: {row['Price Open']}, Close: {row['Price Close']}, High: {row['Price High']}, Low: {row['Price Low']}, Volume: {row['Volume']}", axis=1
                            ),
                            hoverinfo="text",

                    ))
                    fig.show()
                elif chart_type == "Classic Stock Chart":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    quotes = filtered_data[["Date", "Price Open", "Price Close", "Price High", "Price Low"]]
                    westerncandlestick(ax, quotes)
                    ax.set_title(f"Classic Stock Chart: {selected_ticker}")
                    ax.set_xlabel("Volume")
                    ax.set_ylabel("Price")
                    plt.xticks(rotation=45)
                    plt.show()

            except FileNotFoundError:
                print(f"Data file for {selected_ticker} not found.")


dn_dropdown.observe(update_chart, names="value")
ticker_dropdown.observe(update_chart, names="value")
chart_type_dropdown.observe(update_chart, names="value")
timeframe_dropdown.observe(update_chart, names="value")
year_dropdown.observe(update_chart, names="value")

display(widgets.HBox([dn_dropdown, ticker_dropdown, chart_type_dropdown, display_type_dropdown, timeframe_dropdown, year_dropdown]))
display(output)
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from datetime import timedelta
from IPython.display import display, clear_output
import plotly.graph_objects as go
from filterpy.kalman import KalmanFilter
output = widgets.Output()
display(output)
# Add dropdown for technical indicator selection
indicator_dropdown = widgets.Dropdown(
    options=["Select Indicator", "Moving Average (MA)", "Bollinger Bands (BB)", "Relative Strength Index (RSI)",
             "MACD", "Money Flow Index (MFI)","Exponential Moving Average (EMA)", "Kalman Filter"],
    description="Indicator:",
    disabled=False,
    value="Select Indicator"
)

# Add dropdown for technical indicator selection
indicator_dropdown = widgets.Dropdown(
    options=["Select Indicator", "Moving Average (MA)", "Bollinger Bands (BB)", "Relative Strength Index (RSI)",
             "MACD", "Money Flow Index (MFI)","Exponential Moving Average (EMA)", "Kalman Filter"],
    description="Indicator:",
    disabled=False,
)

# Add sliders for technical indicators where necessary
ma_period_dropdown = widgets.Dropdown(
    options=[10, 20, 34, 50, 100],
    value=20,
    description="MA Period:",
    disabled=False,
)

bb_period_dropdown = widgets.Dropdown(
    options=[10, 20, 30, 50],
    value=20,
    description="BB Period:",
    disabled=False,
)

bb_std_dropdown = widgets.Dropdown(
    options=[1.5, 2, 2.5, 3],
    value=2,
    description="BB Std Dev:",
    disabled=False,
)

rsi_period_dropdown = widgets.Dropdown(
    options=[7, 14, 21, 28],
    value=14,
    description="RSI Period:",
    disabled=False,
)

rsi_threshold_dropdown = widgets.Dropdown(
    options=[30, 50, 70],
    value=70,
    description="RSI Threshold:",
    disabled=False,
)

macd_fast_dropdown = widgets.Dropdown(
    options=[9, 12, 15, 20],
    value=12,
    description="MACD Fast:",
    disabled=False,
)

macd_slow_dropdown = widgets.Dropdown(
    options=[26, 30, 40, 50],
    value=26,
    description="MACD Slow:",
    disabled=False,
)

macd_signal_dropdown = widgets.Dropdown(
    options=[7, 9, 12],
    value=9,
    description="MACD Signal:",
    disabled=False,
)

mfi_period_dropdown = widgets.Dropdown(
    options=[7, 14, 21, 28],
    value=14,
    description="MFI Period:",
    disabled=False,
)
ema_period_dropdown = widgets.Dropdown(
    options=[5, 10, 20, 50, 100],
    value=20,
    description="EMA Period:",
    disabled=True,
)
kalman_state_transition_dropdown = widgets.Dropdown(
    options=["Default", "Custom"],
    value="Default",
    description="State Model:",
    disabled=False,
)
kalman_observation_model_dropdown = widgets.Dropdown(
    options=["Default", "Custom"],
    value="Default",
    description="Obs. Model:",
    disabled=False,
)

kalman_process_noise_dropdown = widgets.Dropdown(
    options=["Low", "Medium", "High"],
    value="Medium",
    description="Process Noise:",
    disabled=False,
)

kalman_measurement_noise_dropdown = widgets.Dropdown(
    options=["Low", "Medium", "High"],
    value="Medium",
    description="Meas. Noise:",
    disabled=False,
)

# Display widgets for indicators
indicator_widgets = widgets.VBox([
    indicator_dropdown,
    ma_period_dropdown,
    bb_period_dropdown,
    bb_std_dropdown,
    rsi_period_dropdown,
    rsi_threshold_dropdown,
    macd_fast_dropdown,
    macd_slow_dropdown,
    macd_signal_dropdown,
    mfi_period_dropdown,
    ema_period_dropdown,
    kalman_state_transition_dropdown,
    kalman_observation_model_dropdown,
    kalman_process_noise_dropdown,
    kalman_measurement_noise_dropdown,
])

# Disable all widgets initially
ma_period_dropdown.disabled = True
bb_period_dropdown.disabled = True
bb_std_dropdown.disabled = True
rsi_period_dropdown.disabled = True
rsi_threshold_dropdown.disabled = True
macd_fast_dropdown.disabled = True
macd_slow_dropdown.disabled = True
macd_signal_dropdown.disabled = True
mfi_period_dropdown.disabled = True
ema_period_dropdown.disabled = True
kalman_state_transition_dropdown.disabled = True
kalman_observation_model_dropdown.disabled = True
kalman_process_noise_dropdown.disabled = True
kalman_measurement_noise_dropdown.disabled = True

# Update slider availability based on selected indicator
def update_indicator_dropdown(change):
    selected_indicator = change["new"]

    ma_period_dropdown.disabled = (selected_indicator != "Moving Average (MA)")
    bb_period_dropdown.disabled = (selected_indicator != "Bollinger Bands (BB)")
    bb_std_dropdown.disabled = (selected_indicator != "Bollinger Bands (BB)")
    rsi_period_dropdown.disabled = (selected_indicator != "Relative Strength Index (RSI)")
    rsi_threshold_dropdown.disabled = (selected_indicator != "Relative Strength Index (RSI)")
    macd_fast_dropdown.disabled = (selected_indicator != "MACD")
    macd_slow_dropdown.disabled = (selected_indicator != "MACD")
    macd_signal_dropdown.disabled = (selected_indicator != "MACD")
    mfi_period_dropdown.disabled = (selected_indicator != "Money Flow Index (MFI)")
    ema_period_dropdown.disabled = (selected_indicator != "Exponential Moving Average (EMA)")
    kalman_state_transition_dropdown.disabled = (selected_indicator != "Kalman Filter")
    kalman_observation_model_dropdown.disabled = (selected_indicator != "Kalman Filter")
    kalman_process_noise_dropdown.disabled = (selected_indicator != "Kalman Filter")
    kalman_measurement_noise_dropdown.disabled = (selected_indicator != "Kalman Filter")



indicator_dropdown.observe(update_indicator_dropdown, names="value")

# Update the chart based on selected indicators
def update_chart_with_indicator(change=None):
    with output:
        clear_output()
        selected_ticker = ticker_dropdown.value
        chart_type = chart_type_dropdown.value
        selected_display_type = display_type_dropdown.value
        selected_year = year_dropdown.value
        timeframe = timeframe_dropdown.value
        selected_indicator = indicator_dropdown.value
        indicator_param = None

        if selected_ticker and selected_ticker != "No tickers found":
            file_path = os.path.join(folder_path, f"{selected_ticker}.txt")
            try:
                data = pd.read_csv(file_path, sep="\t", engine="python")
                data["Date"] = pd.to_datetime(data["Date"], errors='coerce')

                if selected_display_type == "Timeframe":
                    end_date = data["Date"].max()
                    if timeframe == "5Y (Monthly)":
                        start_date = end_date - timedelta(days=5 * 365)
                    elif timeframe == "1Y (Monthly)":
                        start_date = end_date - timedelta(days=365)
                    elif timeframe == "1 month (Weekly)":
                        start_date = end_date - timedelta(days=30)
                    filtered_data = data[data["Date"] >= start_date]

                elif selected_display_type == "Year" and selected_year != "Select a ticker first":
                    filtered_data = data[data["Date"].dt.year == int(selected_year)]
                else:
                    filtered_data = pd.DataFrame()

                if filtered_data.empty:
                    print("No data available for the selected filters.")
                    return
                if chart_type == "Candlestick":
                    fig = go.Figure(layout=go.Layout(width=1200, height=900, autosize=True))
                    fig.add_trace(go.Candlestick(
                        x=filtered_data["Date"],
                        open=filtered_data["Price Open"],
                        high=filtered_data["Price High"],
                        low=filtered_data["Price Low"],
                        close=filtered_data["Price Close"],
                    ))

                    # Apply selected technical indicator
                    if selected_indicator == "Moving Average (MA)":
                        ma = calculate_ma(filtered_data, ma_period_dropdown.value)
                        fig.add_trace(go.Scatter(x=filtered_data["Date"], y=ma, mode='lines', name=f"MA ({ma_period_dropdown.value} days)"))
                    elif selected_indicator == "Bollinger Bands (BB)":
                        upper, lower = calculate_bb(filtered_data, bb_period_dropdown.value, bb_std_dropdown.value)
                        fig.add_trace(go.Scatter(x=filtered_data["Date"], y=upper, mode='lines', name=f"BB Upper"))
                        fig.add_trace(go.Scatter(x=filtered_data["Date"], y=lower, mode='lines', name=f"BB Lower"))
                    elif selected_indicator == "Relative Strength Index (RSI)":
                        rsi = calculate_rsi(filtered_data, rsi_period_dropdown.value)
                        fig.add_trace(go.Scatter(x=filtered_data["Date"], y=rsi, mode='lines', name=f"RSI ({rsi_period_dropdown.value} days)"))
                    elif selected_indicator == "MACD":
                        macd, signal = calculate_macd(filtered_data, macd_fast_dropdown.value, macd_slow_dropdown.value, macd_signal_dropdown.value)
                        fig.add_trace(go.Scatter(x=filtered_data["Date"], y=macd, mode='lines', name="MACD"))
                        fig.add_trace(go.Scatter(x=filtered_data["Date"], y=signal, mode='lines', name="MACD Signal"))
                    elif selected_indicator == "Money Flow Index (MFI)":
                        mfi = calculate_mfi(filtered_data, mfi_period_dropdown.value)
                        fig.add_trace(go.Scatter(x=filtered_data["Date"], y=mfi, mode='lines', name=f"MFI ({mfi_period_dropdown.value} days)"))
                    elif selected_indicator == "Exponential Moving Average (EMA)":
                        ema = calculate_ema(filtered_data["Price Close"], ema_period_dropdown.value)
                        fig.add_trace(go.Scatter(x=filtered_data["Date"], y=ema, mode='lines', name=f"EMA ({ema_period_dropdown.value} days)"))
                    elif selected_indicator == "Kalman Filter":
                        kalman_filtered = apply_kalman_filter(filtered_data)
                        fig.add_trace(go.Scatter(x=filtered_data["Date"], y=kalman_filtered, mode='lines', name="Kalman Filter"))


                    fig.show()
                elif chart_type == "Line":
                    fig = fig = go.Figure(layout=go.Layout(width=1200, height=900, autosize=True))
                    fig.add_trace(go.Scatter(
                        x=filtered_data["Date"],
                        y=filtered_data["Price Close"],
                        mode="lines",
                        line=dict(color="blue"),
                        hovertext=filtered_data[["Price Open", "Price Close", "Price High", "Price Low", "Volume"]].apply(
                                lambda row: f"Open: {row['Price Open']}, Close: {row['Price Close']}, High: {row['Price High']}, Low: {row['Price Low']}, Volume: {row['Volume']}", axis=1
                            ),
                            hoverinfo="text",

                    ))
                    fig.show()
                elif chart_type == "Classic Stock Chart":
                    fig, ax = plt.subplots(figsize=(14, 8))
                    quotes = filtered_data[["Date", "Price Open", "Price Close", "Price High", "Price Low"]]
                    westerncandlestick(ax, quotes)
                    ax.set_title(f"Classic Stock Chart: {selected_ticker}")
                    ax.set_xlabel("Volume")
                    ax.set_ylabel("Price")
                    plt.xticks(rotation=45)
                    plt.show()
            except FileNotFoundError:
                print(f"Data file for {selected_ticker} not found.")

# Add observer to the chart update
indicator_dropdown.observe(update_chart_with_indicator, names="value")
ma_period_dropdown.observe(update_chart_with_indicator, names="value")
bb_period_dropdown.observe(update_chart_with_indicator, names="value")
bb_std_dropdown.observe(update_chart_with_indicator, names="value")
rsi_period_dropdown.observe(update_chart_with_indicator, names="value")
rsi_threshold_dropdown.observe(update_chart_with_indicator, names="value")
macd_fast_dropdown.observe(update_chart_with_indicator, names="value")
macd_slow_dropdown.observe(update_chart_with_indicator, names="value")
macd_signal_dropdown.observe(update_chart_with_indicator, names="value")
mfi_period_dropdown.observe(update_chart_with_indicator, names="value")
ema_period_dropdown.observe(update_chart_with_indicator, names="value")
kalman_state_transition_dropdown.observe(update_chart_with_indicator, names="value")
kalman_observation_model_dropdown.observe(update_chart_with_indicator, names="value")
kalman_process_noise_dropdown.observe(update_chart_with_indicator, names="value")
kalman_measurement_noise_dropdown.observe(update_chart_with_indicator, names="value")

# Display the indicator widgets
display(indicator_widgets)

def calculate_ma(data, period):
    """
    Calculate the moving average (MA) for a given period.
    """
    return data["Price Close"].rolling(window=period).mean()

def calculate_bb(data, period, std_dev=2):
    """
    Calculate Bollinger Bands.
    """
    rolling_mean = data["Price Close"].rolling(window=period).mean()
    rolling_std = data["Price Close"].rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band


def calculate_rsi(data, period):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = data["Price Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast_period, slow_period, signal_period=9):
    """
    Calculate MACD and Signal Line.
    """
    ema_fast = data["Price Close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data["Price Close"].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def calculate_mfi(data, period):
    """
    Calculate Money Flow Index (MFI).
    """
    typical_price = (data["Price High"] + data["Price Low"] + data["Price Close"]) / 3
    money_flow = typical_price * data["Volume"]
    positive_flow = money_flow.where(data["Price Close"].diff() > 0, 0)
    negative_flow = money_flow.where(data["Price Close"].diff() < 0, 0)

    positive_mf_sum = positive_flow.rolling(window=period).sum()
    negative_mf_sum = negative_flow.rolling(window=period).sum()

    mfr = positive_mf_sum / negative_mf_sum
    mfi = 100 - (100 / (1 + mfr))
    return mfi
def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average (EMA) for a given period.

    Parameters:
    - data (pd.Series or np.array): Array of closing prices.
    - period (int): The period for the EMA calculation.

    Returns:
    - np.array: EMA values.
    """
    ema = data.ewm(span=period, adjust=False).mean()
    return ema
class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u=None):
        self.x = self.F @ self.x + (self.B @ u if u is not None else 0)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x += K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

def apply_kalman_filter(data):
    """
    Apply Kalman Filter to smooth the price data.
    """
    # Define Kalman Filter parameters
    F = np.array([[1, 1], [0, 1]])  # State transition matrix
    B = np.array([[0], [0]])        # Control matrix (optional)
    H = np.array([[1, 0]])          # Observation matrix
    Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise covariance
    R = np.array([[0.1]])           # Measurement noise covariance
    x0 = np.array([[data["Price Close"].iloc[0]], [0]])  # Initial state
    P0 = np.eye(2)                   # Initial state covariance

    # Initialize the Kalman Filter
    kf = KalmanFilter(F, B, H, Q, R, x0, P0)

    smoothed_prices = []

    # Iterate over the data to apply the Kalman Filter
    for price in data["Price Close"]:
        kf.predict()  # Prediction step
        kf.update(np.array([[price]]))  # Update step
        smoothed_prices.append(kf.x[0, 0])  # Append the smoothed value

    return smoothed_prices

import os
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

def create_stock_signal_analyzer(folder_path="Thailand_done"):
    """
    Creates an interactive stock signal analyzer tool using pandas, and ipywidgets.

    Args:
        folder_path (str, optional): The path to the folder containing the data files.
                                    Defaults to "Thailand_done".

    Returns:
        ipywidgets.VBox: A VBox containing the interactive UI and output area, 
                       or None if there was an error loading the data.
    """

    def load_data(folder_path):
        """Loads data from text files in the specified folder."""
        data_frames = []
        try:
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, file_name)
                    ticker = os.path.splitext(file_name)[0]
                    df = pd.read_csv(
                        file_path,
                        sep="\t",
                        header=None,
                        names=["Date", "Price Open", "Price Low", "Price High", "Price Close", "Volume"]
                    )
                    for col in ["Price Open", "Price Low", "Price High", "Price Close", "Volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df["Ticker"] = ticker
                    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
                    df = df.dropna()
                    data_frames.append(df)
            if not data_frames:
                print(f"Error: No txt files found in '{folder_path}'.")
                return None
            return pd.concat(data_frames, ignore_index=True)
        except FileNotFoundError:
            print(f"Error: Folder not found at '{folder_path}'.")
            return None

    def calculate_indicators(df):
        """Calculates technical indicators."""
        df["Daily Returns"] = df.groupby("Ticker")["Price Close"].pct_change()
        df["EMA_10"] = df.groupby("Ticker")["Price Close"].transform(lambda x: x.ewm(span=10, adjust=False).mean())
        df["EMA_20"] = df.groupby("Ticker")["Price Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        df["Buy Signal"] = (df["Price Close"] > df["EMA_10"]) & (df["EMA_10"] > df["EMA_20"])
        df["Sell Signal"] = (df["Price Close"] < df["EMA_10"]) & (df["EMA_10"] < df["EMA_20"])
        return df
    
    def filter_buy_sell_stocks(df, selected_date):
        """Filters buy/sell stocks based on the selected date."""
        buy_stocks = df[(df["Date"] == selected_date) & (df["Buy Signal"])]
        sell_stocks = df[(df["Date"] == selected_date) & (df["Sell Signal"])]
        return buy_stocks, sell_stocks

    # Load data - Load data only once.
    output_df = load_data(folder_path)
    if output_df is None:
        return None # If there was an error loading the data, return None

    output_df = calculate_indicators(output_df)

    label = widgets.Label(
        value="Xác định các thuộc vùng buy/sell trong ngày:",
        style={
            'font_size': '20px',
            'font_weight': 'bold',
        }
    )

    date_picker = widgets.DatePicker(
        description="Date:",
        disabled=False,
        value=output_df["Date"].min().date(),
    )

    output_area = widgets.Output()

    def update_signals(change=None):
        """Updates buy/sell signals based on the selected date."""
        selected_date = pd.Timestamp(date_picker.value)
        with output_area:
            output_area.clear_output()
            buy_stocks, sell_stocks = filter_buy_sell_stocks(output_df, selected_date)

            if not buy_stocks.empty:
                print(f"Cổ phiếu nằm vùng buy vào ngày {selected_date}:")
                display(buy_stocks[["Ticker", "Price Close", "Date"]])
            else:
                print(f"Không có cổ phiếu nào nằm vùng buy vào ngày {selected_date}.")

            if not sell_stocks.empty:
                print(f"Cổ phiếu nằm vùng sell vào ngày {selected_date}:")
                display(sell_stocks[["Ticker", "Price Close", "Date"]])
            else:
                print(f"Không có cổ phiếu nào nằm vùng sell vào ngày {selected_date}.")


    date_picker.observe(update_signals, names="value")

    ui = widgets.VBox([label, date_picker, output_area])
    return ui

if __name__ == '__main__':
    # Example Usage
    analyzer_ui = create_stock_signal_analyzer()
    if analyzer_ui:
        display(analyzer_ui)

import pandas as pd
import mplcursors
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

def create_stock_rating_visualizer(rs_data_file_path="RS_Scores_with_Color.csv"):
    """
    Creates an interactive stock rating visualizer with zoom and hover using pandas, 
    matplotlib, ipywidgets, and mplcursors.

    Args:
        rs_data_file_path (str, optional): Path to the CSV file containing the RS Score data.
                                           Defaults to "RS_Scores_with_Color.csv".

    Returns:
        ipywidgets.VBox: A VBox containing the UI components and the output area, or None if error happens when loading the CSV file.
    """
    # Load data
    try:
      rs_data = pd.read_csv(rs_data_file_path)
    except FileNotFoundError:
      print(f"Error: File not found at '{rs_data_file_path}'. Please check the file path.")
      return None
    except Exception as e:
      print(f"An unexpected error occurred while loading the CSV file: {e}")
      return None
    
    def plot_selected_tickers(num_tickers, selected_colors):
        """Plots the RS Score for the selected tickers and colors."""

        # If 'All' is selected, use all colors
        if 'All' in selected_colors:
            selected_colors = ["Green", "Yellow", "Red"]

        top_tickers = rs_data.head(num_tickers)["Ticker"].tolist()

        filtered_data = rs_data[
            (rs_data["Ticker"].isin(top_tickers)) &
            (rs_data["Color"].isin(selected_colors))
        ]

        if filtered_data.empty:
          with output_area_chart:
            clear_output(wait=True)
            print("No tickers match the selected criteria.")
          return

        # Create plot
        fig, ax = plt.subplots(figsize=(20, 10))

        # Get colors
        bar_colors = filtered_data["Color"].map({"Green": "green", "Yellow": "yellow", "Red": "red"})

        # plot bars
        bars = ax.bar(filtered_data["Ticker"], filtered_data["RS Score"], color=bar_colors)

        # Separator Lines
        ax.axhline(80, color="green", linestyle="--", label="Leader Zone (RS >= 80)")
        ax.axhline(60, color="yellow", linestyle="--", label="Potential Zone (60 <= RS < 80)")

        # Set titles and labels
        ax.set_title("RS Score Visualization for Selected Stocks", fontsize=16)
        ax.set_xlabel("Ticker", fontsize=12)
        ax.set_ylabel("RS Score", fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend()
        
        # Add hover tooltips
        cursor = mplcursors.cursor(bars, hover=True)
        cursor.connect(
           "add", lambda sel: sel.annotation.set_text(
              f"Ticker: {filtered_data.iloc[sel.index]['Ticker']}\nRS Score: {filtered_data.iloc[sel.index]['RS Score']}"
            )
        )

        #Show plot
        with output_area_chart:
            clear_output(wait=True)
            plt.tight_layout()
            plt.show()

    # Widgets
    num_tickers_slider = widgets.IntSlider(
        value=10, min=1, max=len(rs_data), step=1, description="Top Tickers:"
    )

    color_selection = widgets.SelectMultiple(
        options=["All", "Green", "Yellow", "Red"],
        value=["Green", "Yellow", "Red"],
        description="Colors:",
        rows=3
    )
    
    # Title Label
    title_label = widgets.HTML(value="<h3 style='text-align:center; font-size:20px;'>Stock Rating</h3>")
    
    output_area_chart = widgets.Output()
    
    interactive_plot = widgets.interactive_output(
        plot_selected_tickers,
        {
            "num_tickers": num_tickers_slider,
            "selected_colors": color_selection
        }
    )

    # UI layout
    ui = widgets.VBox([
        title_label,
        widgets.HBox([
          num_tickers_slider,
           color_selection
        ]),
       output_area_chart
    ])
    return ui
    
if __name__ == '__main__':
    # Example usage
    visualizer_ui = create_stock_rating_visualizer()
    if visualizer_ui:
      display(visualizer_ui)