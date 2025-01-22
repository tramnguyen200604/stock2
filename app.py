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

def create_stock_app(folder_path="Thailand_done", file_path='DN.csv'):
    """
    Creates an interactive stock analysis application.

    Args:
        folder_path (str, optional): The path to the folder containing stock data files.
            Defaults to "Thailand_done".
        file_path (str, optional): The path to the CSV file containing company and ticker information.
            Defaults to 'DN.csv'.

    Returns:
        None. Displays the interactive widgets and plots.
    """

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
        options=["5Y (Monthly)", "1Y (Monthly)", "1 month (DaysDays)"],
        description="Timeframe:",
        disabled=True,
    )

    year_dropdown = widgets.Dropdown(
        options=["Select a ticker first"],
        description="Year:",
        disabled=True,
    )
    data_column_dropdown = widgets.Dropdown(
        options=["Price High", "Price Low", "Price Open", "Price Close", "Volume"],
        description="Data Column:",
        disabled=False,
    )


    output = widgets.Output()

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
            selected_column = data_column_dropdown.value
            

            if selected_ticker and selected_ticker != "No tickers found":
                file_path = os.path.join(folder_path, f"{selected_ticker}.txt")
                try:
                    data = pd.read_csv(file_path, sep="\t", engine="python")
                    data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
                    filtered_data = pd.DataFrame()

                    if selected_display_type == "Timeframe" and selected_year != "Select a ticker first":
                        start_year = int(selected_year)
                        end_date = data["Date"].max()
                        start_date = datetime(start_year, 1, 1)

                   
                        if timeframe == "5Y (Monthly)":
                            start_date = end_date - timedelta(days=5 * 365)
                        elif timeframe == "1Y (Monthly)":
                            start_date = end_date - timedelta(days=365)
                        elif timeframe == "1 month (Days)":
                            start_date = end_date - timedelta(days=30)
                        filtered_data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]


                    elif selected_display_type == "Year" and selected_year != "Select a ticker first":
                        filtered_data = data[data["Date"].dt.year == int(selected_year)]
                    else:
                        filtered_data = pd.DataFrame()

                    if filtered_data.empty:
                        print("No data available for the selected filters.")
                        return
                    if selected_column != "Select Data Column":
                        data_to_plot = filtered_data[["Date", selected_column]]
                    else:
                        data_to_plot = filtered_data[["Date", "Price Close"]]
                    

                    if chart_type == "Candlestick":
                        fig = go.Figure(layout=go.Layout(width=1200, height=900, autosize=True))
                        fig.add_trace(go.Candlestick(
                            x=filtered_data["Date"],
                            open=filtered_data["Price Open"],
                            high=filtered_data["Price High"],
                            low=filtered_data["Price Low"],
                            close=filtered_data["Price Close"],
                            increasing_line_color='green',  # Nến tăng màu xanh
                            decreasing_line_color='red',    # Nến giảm màu đỏ
                            name="Candlestick",
                            
                            hovertext=filtered_data[["Price Open", "Price Close", "Price High", "Price Low", "Volume"]].apply(
                                    lambda row: f"Open: {row['Price Open']}, Close: {row['Price Close']}, High: {row['Price High']}, Low: {row['Price Low']}, Volume: {row['Volume']}", axis=1
                                ),
                                hoverinfo="text",

                        ))

                        colors = ['green' if row['Price Close'] > row['Price Open'] else 'red' for _, row in filtered_data.iterrows()]
                        fig.add_trace(go.Bar(
                            x=filtered_data["Date"],
                            y=filtered_data["Volume"],
                            name="Volume",
                            marker_color=colors,
                            yaxis="y2"  # Assign to secondary y-axis
                            ))
                        fig.update_layout(
                            title="Candlestick Chart with Volume",
                            xaxis=dict(
                                title="Date",
                                tickformat="%d-%m-%Y"
                                ),
                            yaxis=dict(title="Price"),
                            yaxis2=dict(
                            title="Volume",
                            overlaying="y",  # Overlay y-axis
                            side="right"     # Place volume on the right
                            ),
                            barmode='relative',  # Keep bars relative
                            height=600
                            )
                        
                        
                        fig.show()
                    elif chart_type == "Line":
                        fig = go.Figure()
                        selected_column = data_column_dropdown.value
                        
                        fig.add_trace(go.Scatter(
                            x=filtered_data["Date"],
                            y=filtered_data[selected_column],
                            mode="lines",
                            line=dict(color="blue"),
                            hovertext=filtered_data[["Price Open", "Price Close", "Price High", "Price Low", "Volume"]].apply(
                                    lambda row: f"Open: {row['Price Open']}, Close: {row['Price Close']}, High: {row['Price High']}, Low: {row['Price Low']}, Volume: {row['Volume']}", axis=1
                                ),
                                hoverinfo="text",

                        ))



                        colors = ['green' if row['Price Close'] > row['Price Open'] else 'red' for _, row in filtered_data.iterrows()]
                        fig.add_trace(go.Bar(
                            x=filtered_data["Date"],
                            y=filtered_data["Volume"],
                            name="Volume",
                            marker_color=colors,  # Semi-transparent blue color for volume bars
                            yaxis="y2"  # This places the volume on a secondary y-axis
                            ))
                        fig.update_layout(
                            title="Line Chart with Volume",
                            xaxis=dict(
                                title="Date",
                                tickformat="%d-%m-%Y"
                            ),
                            yaxis=dict(title="Price"),
                            yaxis2=dict(
                                title="Volume",
                                overlaying="y",  # Overlay y-axis
                                side="right",
                                showgrid=False,
                                range=[0, filtered_data["Volume"].max() * 1.2]  # Set the y-axis range for volume
                            ),
                            barmode='relative',  # Keep bars relative (stacked)
                            height=600
                        )


                        
                        fig.show()
                    elif chart_type == "Classic Stock Chart":
                        colors = filtered_data.apply(lambda row: 'green' if row['Price Close'] > row['Price Open'] else 'red', axis=1)
                        fig = go.Figure(
                             data=[
                                 go.Ohlc(
                                    x=filtered_data['Date'],
                                    open=filtered_data['Price Open'],
                                    high=filtered_data['Price High'],
                                    low=filtered_data['Price Low'],
                                    close=filtered_data['Price Close'],
                                    name='OHLC',
                                    increasing_line_color='green',  # Màu xanh cho nến tăng
                                    decreasing_line_color='red',
                                    hovertext=filtered_data[["Price Open", "Price Close", "Price High", "Price Low", "Volume"]].apply(
                                    lambda row: f"Open: {row['Price Open']}, Close: {row['Price Close']}, High: {row['Price High']}, Low: {row['Price Low']}, Volume: {row['Volume']}", axis=1
                                ),
                                hoverinfo="text",
                                )
                            ]
                        )

                        fig.add_trace(
                            go.Bar(
                                x=filtered_data['Date'],
                                y=filtered_data['Volume'],  # Cột khối lượng giao dịch
                                name='Volume',
                                marker_color=colors,  # Màu sắc khớp với xu hướng giá

                                yaxis='y2'
                            )
                        )
                        fig.update_layout(
                            title='Biểu đồ OHLC với Khối lượng Giao dịch',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            xaxis=dict(
                                 title='Date',
                                tickformat="%d-%m-%Y"  # Set the format to day-month-year
                                 ),
                            yaxis=dict(title='Price', side='left'),
                            yaxis2=dict(
                                title='Volume',
                                overlaying='y',  # Trục phụ cho Volume
                                side='right',
                                range=[0, filtered_data['Volume'].max() * 1.2]
                            ),
                            barmode='relative',  # Đảm bảo thanh Volume độc lập với OHLC
                            xaxis_rangeslider_visible=True,  # Hiển thị thanh trượt
                            #template='plotly_dark',  # Giao diện nền tối
                            height=600,
                        )

                                 
                            
                        fig.show()
                except FileNotFoundError:
                    print(f"Data file for {selected_ticker} not found.")


    dn_dropdown.observe(update_chart, names="value")
    ticker_dropdown.observe(update_chart, names="value")
    chart_type_dropdown.observe(update_chart, names="value")
    display_type_dropdown.observe(update_chart, names="value")

    timeframe_dropdown.observe(update_chart, names="value")
    year_dropdown.observe(update_chart, names="value")
    data_column_dropdown.observe(update_chart, names="value")

    display(widgets.HBox([dn_dropdown, ticker_dropdown, chart_type_dropdown, display_type_dropdown, timeframe_dropdown, year_dropdown, data_column_dropdown]))

    banner = "<div style='display: flex; justify-content: space-between; align-items: center; width: 100%; margin-top: 16px'> <img src='national-flag.jpg' height='150px'/> <div style='color: #081A96; font-size: 75px; font-weight: bold; vertical-align: middle; display: inline-block'> ThaistockWave</div> <img src='city.jpg' width='40%' height='150px' ></div>"

    text = widgets.HTML(banner)

    display(text)

    display(output)

# Example usage:
if __name__ == '__main__':
    create_stock_app()

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
from filterpy.kalman import KalmanFilter
from datetime import datetime, timedelta
import os

def create_indicator_widgets():
    """
    Creates and returns the widgets for technical indicators.

    Returns:
        widgets.VBox: A VBox containing all the indicator widgets.
    """
    output = widgets.Output()
    display(output)

    indicator_dropdown = widgets.Dropdown(
        options=["Select Indicator", "Moving Average (MA)", "Bollinger Bands (BB)", "Relative Strength Index (RSI)",
                "MACD", "Money Flow Index (MFI)", "Exponential Moving Average (EMA)", "Kalman Filter"],
        description="Indicator:",
        disabled=False,
        value="Select Indicator"
    )

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
    return indicator_widgets, indicator_dropdown, ma_period_dropdown, bb_period_dropdown, bb_std_dropdown, rsi_period_dropdown, rsi_threshold_dropdown, macd_fast_dropdown, macd_slow_dropdown, macd_signal_dropdown, mfi_period_dropdown, ema_period_dropdown, kalman_state_transition_dropdown, kalman_observation_model_dropdown, kalman_process_noise_dropdown, kalman_measurement_noise_dropdown, output
def update_chart_with_indicator(folder_path, ticker_dropdown, chart_type_dropdown, display_type_dropdown, year_dropdown, timeframe_dropdown, data_column_dropdown, indicator_dropdown, ma_period_dropdown, bb_period_dropdown, bb_std_dropdown, rsi_period_dropdown, rsi_threshold_dropdown, macd_fast_dropdown, macd_slow_dropdown, macd_signal_dropdown, mfi_period_dropdown, ema_period_dropdown, kalman_state_transition_dropdown, kalman_observation_model_dropdown, kalman_process_noise_dropdown, kalman_measurement_noise_dropdown,output):
    """
    Updates the chart with the selected technical indicator.

    Args:
        folder_path (str): The path to the folder containing stock data files.
        ticker_dropdown (widgets.Dropdown): Dropdown for ticker selection.
        chart_type_dropdown (widgets.Dropdown): Dropdown for chart type selection.
        display_type_dropdown (widgets.Dropdown): Dropdown for display type selection.
        year_dropdown (widgets.Dropdown): Dropdown for year selection.
        timeframe_dropdown (widgets.Dropdown): Dropdown for timeframe selection.
        data_column_dropdown (widgets.Dropdown): Dropdown for data column selection.
        indicator_dropdown (widgets.Dropdown): Dropdown for indicator selection.
        ma_period_dropdown (widgets.Dropdown): Dropdown for MA period.
        bb_period_dropdown (widgets.Dropdown): Dropdown for BB period.
        bb_std_dropdown (widgets.Dropdown): Dropdown for BB standard deviation.
        rsi_period_dropdown (widgets.Dropdown): Dropdown for RSI period.
        rsi_threshold_dropdown (widgets.Dropdown): Dropdown for RSI threshold.
        macd_fast_dropdown (widgets.Dropdown): Dropdown for MACD fast period.
        macd_slow_dropdown (widgets.Dropdown): Dropdown for MACD slow period.
        macd_signal_dropdown (widgets.Dropdown): Dropdown for MACD signal period.
        mfi_period_dropdown (widgets.Dropdown): Dropdown for MFI period.
        ema_period_dropdown (widgets.Dropdown): Dropdown for EMA period.
        kalman_state_transition_dropdown (widgets.Dropdown): Dropdown for Kalman state transition model.
        kalman_observation_model_dropdown (widgets.Dropdown): Dropdown for Kalman observation model.
        kalman_process_noise_dropdown (widgets.Dropdown): Dropdown for Kalman process noise.
        kalman_measurement_noise_dropdown (widgets.Dropdown): Dropdown for Kalman measurement noise.
        output (widgets.Output): Output area for displaying the chart.

    Returns:
        None. Displays the updated chart.
    """
    with output:
        clear_output()
        selected_ticker = ticker_dropdown.value
        chart_type = chart_type_dropdown.value
        selected_display_type = display_type_dropdown.value
        selected_year = year_dropdown.value
        timeframe = timeframe_dropdown.value
        selected_column = data_column_dropdown.value
        selected_indicator = indicator_dropdown.value
        indicator_param = None

        if selected_ticker and selected_ticker != "No tickers found":
            file_path = os.path.join(folder_path, f"{selected_ticker}.txt")
            try:
                data = pd.read_csv(file_path, sep="\t", engine="python")
                data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
                filtered_data = pd.DataFrame()

                # Xử lý theo thời gian hoặc năm
                if selected_display_type == "Timeframe" and selected_year != "Select a ticker first":
                    start_year = int(selected_year)
                    end_date = data["Date"].max()
                    start_date = datetime(start_year, 1, 1)

                    if timeframe == "5Y (Monthly)":
                        start_date = end_date - timedelta(days=5 * 365)
                    elif timeframe == "1Y (Monthly)":
                        start_date = end_date - timedelta(days=365)
                    elif timeframe == "1 month (Days)":
                        start_date = end_date - timedelta(days=30)

                    filtered_data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]

                elif selected_display_type == "Year" and selected_year != "Select a ticker first":
                    filtered_data = data[data["Date"].dt.year == int(selected_year)]
                else:
                    filtered_data = data.copy()

                # Kiểm tra xem dữ liệu có trống không
                if filtered_data.empty:
                    print("No data available for the selected filters.")
                    return

                # Chọn cột dữ liệu để vẽ
                if selected_column != "Select Data Column":
                    data_to_plot = filtered_data[["Date", selected_column]]
                else:
                    data_to_plot = filtered_data[["Date", "Price Close"]]

                # Xử lý vẽ đồ thị
                if chart_type == "Candlestick":
                    fig = go.Figure(layout=go.Layout(width=1200, height=900, autosize=True))
                    fig.add_trace(go.Candlestick(
                        x=filtered_data["Date"],
                        open=filtered_data["Price Open"],
                        high=filtered_data["Price High"],
                        low=filtered_data["Price Low"],
                        close=filtered_data["Price Close"],
                        increasing_line_color='green',  # Màu nến tăng
                        decreasing_line_color='red',    # Màu nến giảm
                        name="Candlestick",
                        hovertext=filtered_data[["Price Open", "Price Close", "Price High", "Price Low", "Volume"]].apply(
                            lambda row: f"Open: {row['Price Open']}, Close: {row['Price Close']}, High: {row['Price High']}, Low: {row['Price Low']}, Volume: {row['Volume']}", axis=1
                        ),
                        hoverinfo="text",
                    ))

                    colors = ['green' if row['Price Close'] > row['Price Open'] else 'red' for _, row in filtered_data.iterrows()]
                    fig.add_trace(go.Bar(
                        x=filtered_data["Date"],
                        y=filtered_data["Volume"],
                        name="Volume",
                        marker_color=colors,
                        yaxis="y2"  # Trục phụ cho volume
                    ))
                    fig.update_layout(
                        title="Candlestick Chart with Volume",
                        xaxis=dict(title="Date", tickformat="%d-%m-%Y"),
                        yaxis=dict(title="Price"),
                        yaxis2=dict(
                            title="Volume",
                            overlaying="y",
                            side="right"
                        ),
                        barmode='relative',  # Giữ bar ở dạng relative
                        height=600
                    )

                    # Áp dụng các chỉ báo kỹ thuật
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
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=filtered_data["Date"],
                        y=filtered_data[selected_column],
                        mode="lines",
                        line=dict(color="blue"),
                        hovertext=filtered_data[["Price Open", "Price Close", "Price High", "Price Low", "Volume"]].apply(
                            lambda row: f"Open: {row['Price Open']}, Close: {row['Price Close']}, High: {row['Price High']}, Low: {row['Price Low']}, Volume: {row['Volume']}", axis=1
                        ),
                        hoverinfo="text",
                    ))

                    colors = ['green' if row['Price Close'] > row['Price Open'] else 'red' for _, row in filtered_data.iterrows()]
                    fig.add_trace(go.Bar(
                        x=filtered_data["Date"],
                        y=filtered_data["Volume"],
                        name="Volume",
                        marker_color=colors,  # Màu sắc cho volume bars
                        yaxis="y2"  # Trục phụ cho volume
                    ))
                    fig.update_layout(
                        title="Line Chart with Volume",
                        xaxis=dict(title="Date", tickformat="%d-%m-%Y"),
                        yaxis=dict(title="Price"),
                        yaxis2=dict(
                            title="Volume",
                            overlaying="y",
                            side="right",
                            showgrid=False,
                            range=[0, filtered_data["Volume"].max() * 1.2]
                        ),
                        barmode='relative',  # Thanh volume
                        height=600
                    )

                    fig.show()

                elif chart_type == "Classic Stock Chart":
                    colors = filtered_data.apply(lambda row: 'green' if row['Price Close'] > row['Price Open'] else 'red', axis=1)
                    fig = go.Figure(
                        data=[
                            go.Ohlc(
                                x=filtered_data['Date'],
                                open=filtered_data['Price Open'],
                                high=filtered_data['Price High'],
                                low=filtered_data['Price Low'],
                                close=filtered_data['Price Close'],
                                name='OHLC',
                                increasing_line_color='green',
                                decreasing_line_color='red',
                                hovertext=filtered_data[["Price Open", "Price Close", "Price High", "Price Low", "Volume"]].apply(
                                    lambda row: f"Open: {row['Price Open']}, Close: {row['Price Close']}, High: {row['Price High']}, Low: {row['Price Low']}, Volume: {row['Volume']}", axis=1
                                ),
                                hoverinfo="text",
                            )
                        ]
                    )

                    fig.add_trace(
                        go.Bar(
                            x=filtered_data['Date'],
                            y=filtered_data['Volume'],
                            name='Volume',
                            marker_color=colors,
                            yaxis='y2'
                        )
                    )
                    fig.update_layout(
                        title='Biểu đồ OHLC với Khối lượng Giao dịch',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        xaxis=dict(
                            title='Date',
                            tickformat="%d-%m-%Y"
                        ),
                        yaxis=dict(title='Price', side='left'),
                        yaxis2=dict(
                            title='Volume',
                            overlaying='y',
                            side='right',
                            range=[0, filtered_data['Volume'].max() * 1.2]
                        ),
                        barmode='relative',
                        xaxis_rangeslider_visible=True,
                        height=600,
                    )

                    fig.show()

            except FileNotFoundError:
                print(f"Data file for {selected_ticker} not found.")

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

if __name__ == '__main__':
    from create_stock_app import create_stock_app
    folder_path = "Thailand_done"
    app = create_stock_app(folder_path=folder_path)
    indicator_widgets, indicator_dropdown, ma_period_dropdown, bb_period_dropdown, bb_std_dropdown, rsi_period_dropdown, rsi_threshold_dropdown, macd_fast_dropdown, macd_slow_dropdown, macd_signal_dropdown, mfi_period_dropdown, ema_period_dropdown, kalman_state_transition_dropdown, kalman_observation_model_dropdown, kalman_process_noise_dropdown, kalman_measurement_noise_dropdown, output = create_indicator_widgets()
    display(indicator_widgets)

    def observe_indicator_changes(change):
        update_chart_with_indicator(folder_path, app[1], app[2], app[3], app[4], app[5], app[6], indicator_dropdown, ma_period_dropdown, bb_period_dropdown, bb_std_dropdown, rsi_period_dropdown, rsi_threshold_dropdown, macd_fast_dropdown, macd_slow_dropdown, macd_signal_dropdown, mfi_period_dropdown, ema_period_dropdown, kalman_state_transition_dropdown, kalman_observation_model_dropdown, kalman_process_noise_dropdown, kalman_measurement_noise_dropdown, output)
    indicator_dropdown.observe(observe_indicator_changes, names="value")
    ma_period_dropdown.observe(observe_indicator_changes, names="value")
    bb_period_dropdown.observe(observe_indicator_changes, names="value")
    bb_std_dropdown.observe(observe_indicator_changes, names="value")
    rsi_period_dropdown.observe(observe_indicator_changes, names="value")
    rsi_threshold_dropdown.observe(observe_indicator_changes, names="value")
    macd_fast_dropdown.observe(observe_indicator_changes, names="value")
    macd_slow_dropdown.observe(observe_indicator_changes, names="value")
    macd_signal_dropdown.observe(observe_indicator_changes, names="value")
    mfi_period_dropdown.observe(observe_indicator_changes, names="value")
    ema_period_dropdown.observe(observe_indicator_changes, names="value")
    kalman_state_transition_dropdown.observe(observe_indicator_changes, names="value")
    kalman_observation_model_dropdown.observe(observe_indicator_changes, names="value")
    kalman_process_noise_dropdown.observe(observe_indicator_changes, names="value")
    kalman_measurement_noise_dropdown.observe(observe_indicator_changes, names="value")

import os
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from scipy.signal import lfilter


def create_buy_sell_analysis_app(folder_path="Thailand_done"):
    """
    Creates an interactive application to analyze buy/sell signals for stocks.

    Args:
        folder_path (str, optional): The path to the folder containing stock data files.
            Defaults to "Thailand_done".

    Returns:
        None. Displays the interactive widgets and results.
    """

    # ============================= Helper Functions =============================
    def load_data_v2(folder_path):
        data_frames = []
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
                # Chuyển đổi dữ liệu thành số
                for col in ["Price Open", "Price Low", "Price High", "Price Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["Ticker"] = ticker
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna()
                data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True)

    def calculate_daily_returns_v2(df):
        df["Daily Returns"] = df.groupby("Ticker")["Price Close"].pct_change()

    def calculate_volume_ratio_v2(df):
        total_volume = df.groupby("Ticker")["Volume"].transform("sum")
        df["Volume Ratio"] = df["Volume"] / total_volume

    def calculate_ema_v2(df, column="Price Close", spans=[10, 20]):
        for span in spans:
            df[f"EMA_{span}"] = df.groupby("Ticker")[column].transform(lambda x: x.ewm(span=span, adjust=False).mean())

    def calculate_rsi_v2(df, column="Price Close", window=14):
        def rsi(series):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df[f"RSI_{window}"] = df.groupby("Ticker")[column].transform(rsi)

    def apply_kalman_filter_v2(df, column="Price Close"):
        def kalman_filter(series):
            n = len(series)
            z = np.array(series)
            xhat = np.zeros(n)
            P = np.zeros(n)
            xhatminus = np.zeros(n)
            Pminus = np.zeros(n)
            K = np.zeros(n)

            Q = 1e-5  # process variance
            R = 0.01  # measurement variance

            xhat[0] = z[0]
            P[0] = 1.0

            for k in range(1, n):
                xhatminus[k] = xhat[k - 1]
                Pminus[k] = P[k - 1] + Q
                K[k] = Pminus[k] / (Pminus[k] + R)
                xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
                P[k] = (1 - K[k]) * Pminus[k]

            return pd.Series(xhat)

        df["Kalman"] = df.groupby("Ticker")[column].transform(kalman_filter)

    def calculate_bollinger_bands_v2(df, column="Price Close", window=20):
        rolling_mean = df.groupby("Ticker")[column].transform(lambda x: x.rolling(window).mean())
        rolling_std = df.groupby("Ticker")[column].transform(lambda x: x.rolling(window).std())
        df["BB_Mid"] = rolling_mean
        df["BB_Up2"] = rolling_mean + (rolling_std * 2)
        df["BB_Low2"] = rolling_mean - (rolling_std * 2)
        df["BB_Up"] = rolling_mean + (rolling_std * 1.5)
        df["BB_Low"] = rolling_mean - (rolling_std * 1.5)

    def calculate_macd_v2(df, column="Price Close", short_span=12, long_span=26, signal_span=9):
        df["EMA_short"] = df.groupby("Ticker")[column].transform(lambda x: x.ewm(span=short_span, adjust=False).mean())
        df["EMA_long"] = df.groupby("Ticker")[column].transform(lambda x: x.ewm(span=long_span, adjust=False).mean())
        df["MACD"] = df["EMA_short"] - df["EMA_long"]
        df["Signal"] = df.groupby("Ticker")["MACD"].transform(lambda x: x.ewm(span=signal_span, adjust=False).mean())

    def calculate_ma_v2(df, column="Price Close", window=50):
        df[f"MA_{window}"] = df.groupby("Ticker")[column].transform(lambda x: x.rolling(window).mean())

    def calculate_all_indicators_v2(df):
        calculate_daily_returns_v2(df)
        calculate_volume_ratio_v2(df)
        calculate_ema_v2(df, spans=[10, 20])
        calculate_ma_v2(df, window=50)
        calculate_rsi_v2(df, window=14)
        apply_kalman_filter_v2(df)
        calculate_bollinger_bands_v2(df)
        calculate_macd_v2(df)
    
    def generate_buy_sell_signal(df):
         # Tín hiệu mua (Buy Signal):
        df["Buy Signal"] = np.where(
            (df["Daily Returns"] > 0.02) &  # Tỷ suất lợi nhuận hằng ngày > 2%
            (df["RSI_14"] > 30) &  # RSI > 30
            (df["Volume Ratio"] > df["Volume Ratio"].median()) &  # Tỷ lệ khối lượng giao dịch > trung vị
            (df["MACD"] > df["Signal"]),  # MACD cắt lên đường tín hiệu
            "Buy",  # Tín hiệu mua
            np.nan  # Không có tín hiệu
        )

        # Tín hiệu bán (Sell Signal):
        df["Sell Signal"] = np.where(
            (df["Daily Returns"] < -0.02) &  # Tỷ suất lợi nhuận hằng ngày < -2%
            (df["RSI_14"] > 70) &  # RSI > 70
            (df["Volume Ratio"] > df["Volume Ratio"].median()) &  # Tỷ lệ khối lượng giao dịch > trung vị
            (df["MACD"] < df["Signal"]),  # MACD cắt xuống đường tín hiệu
            "Sell",  # Tín hiệu bán
            np.nan  # Không có tín hiệu
        )
    
        # EMA 10 và EMA 20: Tín hiệu kháng cự giữa EMA 10 và EMA 20
        df["Buy Signal"] = np.where(
          (df["EMA_10"] < df["EMA_20"]) &  # EMA 10 thấp hơn EMA 20 (kháng cự)
          (df["Price Close"] > df["EMA_10"]) & (df["Price Close"] < df["EMA_20"]),  # Giá chạm vào vùng giữa EMA 10 và EMA 20
          "Sell",  # Tín hiệu bán
          df["Buy Signal"]  # Giữ tín hiệu cũ
        )
        
        return df


    def filter_buy_sell_stocks(df):
        buy_stocks = df[df["Buy Signal"] == "Buy"]
        sell_stocks = df[df["Sell Signal"] == "Sell"]
        return buy_stocks, sell_stocks

    # ============================= Main Logic =============================
    output_df = load_data_v2(folder_path)

    # Clean the data
    essential_columns = ["Ticker", "Date", "Price Open", "Price Low", "Price High", "Price Close", "Volume"]
    output_df = output_df.dropna(subset=essential_columns)
    output_df["Date"] = pd.to_datetime(output_df["Date"], errors="coerce")
    output_df = output_df.dropna(subset=["Date"])
    numeric_columns = ["Price Open", "Price Low", "Price High", "Price Close", "Volume"]
    output_df[numeric_columns] = output_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    output_df = output_df.dropna(subset=numeric_columns)
    output_df = output_df.drop_duplicates()
    
    calculate_all_indicators_v2(output_df)
    output_df = generate_buy_sell_signal(output_df)
    buy_stocks, sell_stocks = filter_buy_sell_stocks(output_df)
    
    # Xác định ngày cần lọc
    selected_date = "2022-12-29"
    # Lọc ra danh sách cổ phiếu cần mua trong ngày selected_date
    buy_list = output_df[(output_df["Date"] == selected_date) & (output_df["Buy Signal"] == "Buy")]

    # Lọc ra danh sách cổ phiếu cần bán trong ngày selected_date
    sell_list = output_df[(output_df["Date"] == selected_date) & (output_df["Sell Signal"] == "Sell")]
    
    # Output area
    output = widgets.Output()
    display(output)

    with output:
        clear_output()
        # Style for table
        table_style = """
        <style>
            .dataframe {
                border-collapse: collapse;
            }
            .dataframe th, .dataframe td {
                border: 1px solid black;
                padding: 5px;
                text-align: center;
            }
            .dataframe th {
                background-color: #f0f0f0;
                font-weight: bold;
            }
        </style>
        """
        
        # Define HTML title using widgets.HTML
        display(widgets.HTML(value="<h2>Cổ phiếu nên mua/bán trong ngày</h2>"))
        
         # Display buy stocks with style
        if not buy_stocks.empty:
           display(widgets.HTML(value=f"<h3>Cổ phiếu nên mua trong ngày</h3>"))
           buy_list_table_html = table_style + buy_list[["Ticker", "Date", "Price Close", "Buy Signal"]].style.set_table_attributes('class="dataframe"').to_html()
           display(widgets.HTML(value=buy_list_table_html))

        else:
            display(widgets.HTML(value="<h3>Không có cổ phiếu nào được khuyến nghị mua</h3>"))
        
        # Display sell stocks with style
        if not sell_stocks.empty:
           display(widgets.HTML(value=f"<br><h3>Cổ phiếu nên bán trong ngày</h3>"))
           sell_list_table_html = table_style + sell_list[["Ticker", "Date", "Price Close", "Sell Signal"]].style.set_table_attributes('class="dataframe"').to_html()
           display(widgets.HTML(value=sell_list_table_html))

        else:
            display(widgets.HTML(value="<h3>Không có cổ phiếu nào được khuyến nghị bán</h3>"))
if __name__ == '__main__':
    create_buy_sell_analysis_app()

import pandas as pd
import mplcursors
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

def create_rs_score_visualization(rs_data_file_path="RS_Scores_with_Color.csv"):
    """
    Creates an interactive visualization for RS scores of stocks.

    Args:
        rs_data_file_path (str, optional): The path to the CSV file containing RS score data.
            Defaults to "RS_Scores_with_Color.csv".

    Returns:
        None. Displays the interactive widgets and plot.
    """

    # Đọc dữ liệu từ file RS_Scores_with_Color.csv
    rs_data = pd.read_csv(rs_data_file_path)

    # Hàm vẽ biểu đồ với tính năng zoom
    def plot_selected_tickers(num_tickers, selected_colors):
        # Nếu chọn 'All', thì lấy tất cả các màu
        if 'All' in selected_colors:
            selected_colors = ["Green", "Yellow", "Red"]

        # Lấy danh sách Top N cổ phiếu theo số lượng người dùng chọn
        top_tickers = rs_data.head(num_tickers)["Ticker"].tolist()

        # Lọc dữ liệu theo danh sách mã cổ phiếu và màu sắc
        filtered_data1 = rs_data[
            (rs_data["Ticker"].isin(top_tickers)) &
            (rs_data["Color"].isin(selected_colors))
        ]

        if filtered_data1.empty:
            clear_output(wait=True)
            print("Không có mã cổ phiếu nào phù hợp với lựa chọn.")
            return

        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=(20, 10))

        # Lấy màu sắc từ cột "Color"
        bar_colors = filtered_data1["Color"].map({"Green": "green", "Yellow": "yellow", "Red": "red"})

        # Vẽ biểu đồ cột
        ax.bar(filtered_data1["Ticker"], filtered_data1["RS Score"], color=bar_colors)

        # Đường phân cách các vùng màu sắc
        ax.axhline(80, color="green", linestyle="--", label="Leader Zone (RS >= 80)")
        ax.axhline(60, color="yellow", linestyle="--", label="Potential Zone (60 <= RS < 80)")

        # Thiết lập tiêu đề và nhãn
        ax.set_title("RS Score Visualization for Selected Stocks", fontsize=16)
        ax.set_xlabel("Ticker", fontsize=12)
        ax.set_ylabel("RS Score", fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend()

        # Thêm tính năng tương tác (zoom và chọn)
        cursor = mplcursors.cursor(ax.bar(filtered_data1["Ticker"], filtered_data1["RS Score"], color=bar_colors), hover=True)
        cursor.connect(
            "add", lambda sel: sel.annotation.set_text(
                f"Ticker: {filtered_data1.iloc[sel.index]['Ticker']}\nRS Score: {filtered_data1.iloc[sel.index]['RS Score']}"
            )
        )

        # Hiển thị biểu đồ
        clear_output(wait=True)
        plt.tight_layout()
        plt.show()

    # Widget chọn số lượng cổ phiếu hiển thị
    num_tickers_slider = widgets.IntSlider(
        value=10, min=1, max=len(rs_data), step=1, description="Top Tickers:"
    )

    # Widget chọn màu với SelectMultiple
    color_selection = widgets.SelectMultiple(
        options=["All", "Green", "Yellow", "Red"],  # Thêm lựa chọn "All"
        value=["Green", "Yellow", "Red"],  # Mặc định chọn tất cả màu
        description="Colors:",
        rows=3
    )

    # Hiển thị tiêu đề "Stock Rating"
    title_label = widgets.HTML(value="<h3 style='text-align:center; font-size:20px;'>Stock Rating</h3>")

    # Output area để hiển thị biểu đồ
    output_area_chart = widgets.Output()

    # Kết hợp widgets và hàm cập nhật với interactive
    interactive_plot = widgets.interactive_output(
        plot_selected_tickers,
        {
            "num_tickers": num_tickers_slider,
            "selected_colors": color_selection
        }
    )

    # Hiển thị giao diện người dùng
    ui = widgets.VBox([
        title_label,
        widgets.HBox([num_tickers_slider, color_selection]),
        output_area_chart
    ])

    # Hiển thị giao diện và biểu đồ
    display(ui, interactive_plot)

if __name__ == '__main__':
    create_rs_score_visualization()

import os
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

def create_stock_recommendation_app(folder_path="Thailand_done", dn_file_path='DN_filtered.csv', excel_path='Thailand.xlsx', rs_scores_path='RS_Scores.csv'):
    """
    Creates an interactive application to display stock recommendations based on various factors.

    Args:
        folder_path (str, optional): The path to the folder containing stock data files.
            Defaults to "Thailand_done".
        dn_file_path (str, optional): The path to the CSV file containing DN (company) data.
            Defaults to 'DN_filtered.csv'.
        excel_path (str, optional): The path to the Excel file containing additional stock info.
            Defaults to 'Thailand.xlsx'.
        rs_scores_path (str, optional): The path to the CSV file containing RS scores.
            Defaults to 'RS_Scores.csv'.

    Returns:
        None. Displays the interactive widgets and output.
    """

    # ============================= Helper Functions =============================
    def load_data_v2(folder_path):
        data_frames = []
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
                # Chuyển đổi dữ liệu thành số
                for col in ["Price Open", "Price Low", "Price High", "Price Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["Ticker"] = ticker
                df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
                df = df.dropna()  # Loại bỏ các hàng chứa giá trị không hợp lệ
                data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True)

    def calculate_daily_returns_v2(df):
        df["Daily Returns"] = df.groupby("Ticker")["Price Close"].pct_change()

    def calculate_volume_ratio_v2(df):
        total_volume = df.groupby("Ticker")["Volume"].transform("sum")
        df["Volume Ratio"] = df["Volume"] / total_volume

    def calculate_ema_v2(df, column="Price Close", spans=[10, 20]):
        for span in spans:
            df[f"EMA_{span}"] = df.groupby("Ticker")[column].transform(lambda x: x.ewm(span=span, adjust=False).mean())

    def calculate_rsi_v2(df, column="Price Close", window=14):
        def rsi(series):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df[f"RSI_{window}"] = df.groupby("Ticker")[column].transform(rsi)
    
    def apply_kalman_filter_v2(df, column="Price Close"):
        def kalman_filter(series):
            n = len(series)
            z = np.array(series)
            xhat = np.zeros(n)
            P = np.zeros(n)
            xhatminus = np.zeros(n)
            Pminus = np.zeros(n)
            K = np.zeros(n)

            Q = 1e-5  # process variance
            R = 0.01  # measurement variance

            xhat[0] = z[0]
            P[0] = 1.0

            for k in range(1, n):
                xhatminus[k] = xhat[k - 1]
                Pminus[k] = P[k - 1] + Q
                K[k] = Pminus[k] / (Pminus[k] + R)
                xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
                P[k] = (1 - K[k]) * Pminus[k]

            return pd.Series(xhat)

        df["Kalman"] = df.groupby("Ticker")[column].transform(kalman_filter)

    def calculate_bollinger_bands_v2(df, column="Price Close", window=20):
        rolling_mean = df.groupby("Ticker")[column].transform(lambda x: x.rolling(window).mean())
        rolling_std = df.groupby("Ticker")[column].transform(lambda x: x.rolling(window).std())
        df["BB_Mid"] = rolling_mean
        df["BB_Up2"] = rolling_mean + (rolling_std * 2)
        df["BB_Low2"] = rolling_mean - (rolling_std * 2)
        df["BB_Up"] = rolling_mean + (rolling_std * 1.5)
        df["BB_Low"] = rolling_mean - (rolling_std * 1.5)

    def calculate_macd_v2(df, column="Price Close", short_span=12, long_span=26, signal_span=9):
      df["EMA_short"] = df.groupby("Ticker")[column].transform(lambda x: x.ewm(span=short_span, adjust=False).mean())
      df["EMA_long"] = df.groupby("Ticker")[column].transform(lambda x: x.ewm(span=long_span, adjust=False).mean())
      df["MACD"] = df["EMA_short"] - df["EMA_long"]
      df["Signal"] = df.groupby("Ticker")["MACD"].transform(lambda x: x.ewm(span=signal_span, adjust=False).mean())
    
    def calculate_ma_v2(df, column="Price Close", window=50):
      df[f"MA_{window}"] = df.groupby("Ticker")[column].transform(lambda x: x.rolling(window).mean())

    def calculate_all_indicators(df):
        calculate_daily_returns_v2(df)
        calculate_volume_ratio_v2(df)
        calculate_ema_v2(df, spans=[10, 20])
        calculate_ma_v2(df, window=50)
        calculate_rsi_v2(df, window=14)
        apply_kalman_filter_v2(df)
        calculate_bollinger_bands_v2(df)
        calculate_macd_v2(df)

    def generate_buy_sell_signal(df):
      # Tín hiệu mua (Buy Signal):
        df["Buy Signal"] = np.where(
            (df["Daily Returns"] > 0.02) &  # Tỷ suất lợi nhuận hằng ngày > 2%
            (df["RSI_14"] > 30) &  # RSI > 30
            (df["Volume Ratio"] > df["Volume Ratio"].median()) &  # Tỷ lệ khối lượng giao dịch > trung vị
            (df["MACD"] > df["Signal"]),  # MACD cắt lên đường tín hiệu
            "Buy",  # Tín hiệu mua
            np.nan  # Không có tín hiệu
        )

        # Tín hiệu bán (Sell Signal):
        df["Sell Signal"] = np.where(
            (df["Daily Returns"] < -0.02) &  # Tỷ suất lợi nhuận hằng ngày < -2%
            (df["RSI_14"] > 65) &  # RSI > 65
            (df["Volume Ratio"] > df["Volume Ratio"].median()) &  # Tỷ lệ khối lượng giao dịch > trung vị
            (df["MACD"] < df["Signal"]),  # MACD cắt xuống đường tín hiệu
            "Sell",  # Tín hiệu bán
            np.nan  # Không có tín hiệu
        )

         # EMA 10 và EMA 20: Tín hiệu kháng cự giữa EMA 10 và EMA 20
        df["Buy Signal"] = np.where(
            (df["EMA_10"] < df["EMA_20"]) &  # EMA 10 thấp hơn EMA 20 (kháng cự)
            (df["Price Close"] > df["EMA_10"]) & (df["Price Close"] < df["EMA_20"]),  # Giá chạm vào vùng giữa EMA 10 và EMA 20
            "Sell",  # Tín hiệu bán
            df["Buy Signal"]  # Giữ tín hiệu cũ
        )

        return df
    def filter_buy_sell_stock(df):
    # Lọc các cổ phiếu có tín hiệu mua
        buy_stocks = output_df[output_df["Buy Signal"] == "Buy"]

    # Lọc các cổ phiếu có tín hiệu bán
        sell_stocks = output_df[output_df["Sell Signal"] == "Sell"]

        return buy_stocks, sell_stocks

    # ============================= Main Logic =============================
    # Load data
    output_df = load_data_v2(folder_path)
    
    # Process data
    calculate_all_indicators(output_df)
    output_df = generate_buy_sell_signal(output_df)
    buy_stocks, sell_stocks = filter_buy_sell_stock(output_df)
    
    # Load RS Scores
    rs_scores_df = pd.read_csv(rs_scores_path)

    # Load data from folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    data_frames = []
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        ticker = os.path.splitext(file_name)[0]
        try:
            df = pd.read_csv(file_path, sep="\t", header=None)
            if df.shape[1] != 6:
                continue
            df = pd.read_csv(
                file_path,
                sep="\t",
                header=0,
                names=["Date", "Price Open", "Price Low", "Price High", "Price Close", "Volume"]
            )
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Ticker"] = ticker
            data_frames.append(df)

        except Exception:
            pass
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df = combined_df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
    # Load DN data
    df_dn = pd.read_csv(dn_file_path)
    combined_df = combined_df.merge(df_dn, how="left", left_on="Ticker", right_on="RIC")

    # Load excel data
    df_excel = pd.read_excel(excel_path)
    df_excel['RIC_Prefix'] = df_excel['RIC'].str.split('.').str[0]
    combined_df = combined_df.merge(rs_scores_df, how="left", left_on="Ticker", right_on="Ticker")
    combined_df = combined_df.merge(df_excel, how="left", left_on="Ticker", right_on="RIC_Prefix")

    # Filter data for specific date
    selected_date = pd.to_datetime("2022-12-29")
    df_filtered = combined_df[combined_df['Date'] == selected_date]
    tickers = df_filtered['Ticker'].unique().tolist()
    
    # Widgets
    dropdown = widgets.Dropdown(
        options=['Chọn cổ phiếu'] + tickers,
        description='Cổ phiếu:'
    )
    
    text_input = widgets.Text(
        value='',
        placeholder='Nhập mã cổ phiếu',
        description='Mã cổ phiếu:'
    )

    input_box = widgets.HBox([dropdown, text_input])

    output1 = widgets.Output()

    def show_stock_info(change):
      with output1:
        clear_output()
        selected_stock = dropdown.value if dropdown.value != 'Chọn cổ phiếu' else text_input.value.strip()

        if selected_stock:
            if selected_stock in df_filtered['Ticker'].values:
                stock_data = df_filtered[df_filtered['Ticker'] == selected_stock].iloc[0]

                # Hiển thị tín hiệu mua và bán dưới dạng bảng kiểu mới
                selected_date = "2022-12-29"
                buy_list = output_df[(output_df["Date"] == selected_date) & (output_df["Buy Signal"] == "Buy")]
                sell_list = output_df[(output_df["Date"] == selected_date) & (output_df["Sell Signal"] == "Sell")]

                # Hiển thị hai bảng tín hiệu mua và bán phía trên
                display(widgets.HTML(value=f"""
                <div style="display: flex; flex-direction: row; justify-content: space-between; margin-bottom: 20px;">
                    <div style="flex: 1; background-color: #1f1f1f; color: #00cc66; border: 2px solid #00cc66; border-radius: 8px; padding: 10px; margin-right: 10px;">
                        <h4 style="margin: 0; padding: 0; display: flex; align-items: center; color: #00cc66;">
                            <span style="margin-right: 10px;">☀</span> TÍN HIỆU MUA
                        </h4>
                        <div style="display: flex; flex-wrap: wrap; margin-top: 10px;">
                            {''.join([f'<div style="margin: 5px; padding: 10px 15px; background-color: #333; color: #fff; border-radius: 4px;">{ticker}</div>' for ticker in buy_list["Ticker"].tolist()])}
                        </div>
                    </div>
                    <div style="flex: 1; background-color: #1f1f1f; color: #cc3300; border: 2px solid #cc3300; border-radius: 8px; padding: 10px; margin-left: 10px;">
                        <h4 style="margin: 0; padding: 0; display: flex; align-items: center; color: #cc3300;">
                            <span style="margin-right: 10px;">⛅</span> TÍN HIỆU BÁN
                        </h4>
                        <div style="display: flex; flex-wrap: wrap; margin-top: 10px;">
                            {''.join([f'<div style="margin: 5px; padding: 10px 15px; background-color: #333; color: #fff; border-radius: 4px;">{ticker}</div>' for ticker in sell_list["Ticker"].tolist()])}
                        </div>
                    </div>
                </div>
                """))

                # Hiển thị bảng mã cổ phiếu và tên công ty
                display(widgets.HTML(value=f"""
                <div style="display: flex; flex-direction: row; justify-content: space-between; background-color: #1f1f1f; color: #fff; padding: 10px; border-radius: 8px; margin-bottom: 20px; gap: 20px;">
                    <div style="flex: 1; text-align: left; padding: 10px;">
                        <h4 style="margin: 0; font-size: 18px; color: #fff;">MÃ CP</h4>
                        <p style="margin: 5px 0; font-size: 16px; font-weight: bold; color: #fff;">{stock_data['Ticker']}</p>
                    </div>
                    <div style="flex: 2; text-align: left; padding: 10px; border-left: 1px solid #444; padding-left: 10px;">
                        <h4 style="margin: 0; font-size: 18px; color: #fff;">TÊN CÔNG TY</h4>
                        <p style="margin: 5px 0; font-size: 16px; font-weight: bold; color: #fff;">{stock_data['Name']}</p>
                    </div>
                </div>
                """))

                # Tính tín hiệu giao dịch dựa trên buy_list và sell_list
                if (selected_stock in buy_list['Ticker'].tolist()) and (buy_list[buy_list['Ticker'] == selected_stock]['Buy Signal'].iloc[0] == "Buy"):
                    signal = 'Mua'
                    signal_color = '#00cc66'  # Xanh lá cây
                elif (selected_stock in sell_list['Ticker'].tolist()) and (sell_list[sell_list['Ticker'] == selected_stock]['Sell Signal'].iloc[0] == "Sell"):
                    signal = 'Bán'
                    signal_color = '#cc3300'  # Đỏ
                else:
                    signal = 'Giữ'
                    signal_color = '#ffd700'  # Vàng

                # Lấy Stock Rating và màu sắc từ rs_scores_df
                stock_rating = stock_data['RS Score']
                stock_color = stock_data['Color'] if 'Color' in stock_data else '#444'

                # Hiển thị thông tin theo định dạng bảng
                display(widgets.HTML(value=f"""
                <div style="display: flex; flex-direction: row; background-color: #1f1f1f; color: #fff; padding: 20px; border-radius: 8px; align-items: flex-start; margin-top: 20px; gap: 20px;">
                    <div style="flex: 1; text-align: center; display: flex; flex-direction: column; justify-content: flex-start; align-items: center;">
                        <h4 style="color: #00cc66; margin-bottom: 30px; font-size: 20px;">KHUYẾN NGHỊ</h4>
                        <div style="width: 150px; height: 150px; background-color: {signal_color}; color: #fff; border-radius: 8px; display: flex; justify-content: center; align-items: center; font-size: 32px; font-weight: bold;">
                            {signal}
                        </div>
                    </div>
                    <div style="flex: 1; text-align: center; display: flex; flex-direction: column; justify-content: flex-start; align-items: center; border-left: 1px solid #444; padding-left: 10px;">
                        <h4 style="color: #00cc66; margin-bottom: 30px; font-size: 20px;">STOCK RATING</h4>
                        <div style="width: 150px; height: 150px; background-color: {stock_color}; color: #fff; border-radius: 8px; display: flex; justify-content: center; align-items: center; font-size: 32px; font-weight: bold;">
                            {stock_rating}
                        </div>
                    </div>
                    <div style="flex: 2; text-align: left; border-left: 1px solid #444; padding-left: 10px;">
                        <h4 style="color: #00cc66; margin-bottom: 30px; font-size: 20px;">THÔNG TIN CƠ BẢN</h4>
                        <p style="margin: 15px 0;">- Nhóm Ngành: {stock_data['Sector']}</p>
                        <p style="margin: 15px 0;">- Ngành: {stock_data['Full Name']}</p>
                        <p style="margin: 15px 0;">- GTGD: {stock_data['Volume']}</p>
                        <p style="margin: 15px 0;">- KLGD (CP): {stock_data['Price High']}</p>
                        <p style="margin: 15px 0;">- Thị trường giao dịch: {stock_data['Market']}</p>
                        <p style="margin: 15px 0;">- Tỷ suất cổ tức: {stock_data['RS Score']}%</p>
                    </div>
                    <div style="flex: 2; text-align: left; border-left: 1px solid #444; padding-left: 10px;">
                        <h4 style="color: #00cc66; margin-bottom: 30px; font-size: 20px;">THÔNG TIN XU HƯỚNG</h4>
                        <p style="margin: 15px 0;">- Giá mở cửa: {stock_data['Price Open']}</p>
                        <p style="margin: 15px 0;">- Giá thấp nhất: {stock_data['Price Low']}</p>
                        <p style="margin: 15px 0;">- Giá cao nhất: {stock_data['Price High']}</p>
                        <p style="margin: 15px 0;">- Giá đóng cửa: {stock_data['Price Close']}</p>
                        <p style="margin: 15px 0;">- Khối lượng: {stock_data['Volume']}</p>
                        <p style="margin: 15px 0;">- Price Change (%): {stock_data['Price Change (%)']:.2f}%</p>
                        <p style="margin: 15px 0;">- RS Score: {stock_data['RS Score']}</p>
                    </div>
                </div>
                """))

            else:
                print(f"Không tìm thấy cổ phiếu {selected_stock} trong dữ liệu.")
        else:
            print("Vui lòng chọn hoặc nhập mã cổ phiếu.")

    # Gắn hàm với dropdown và text input
    dropdown.observe(show_stock_info, names='value')
    text_input.observe(show_stock_info, names='value')

    title_widget = widgets.HTML(value="<h2>Vui lòng chọn hoặc nhập mã cổ phiếu để xem khuyến nghị</h2>")

    # Hiển thị HBox, output
    display(title_widget, input_box, output1)
if __name__ == '__main__':
    create_stock_recommendation_app()