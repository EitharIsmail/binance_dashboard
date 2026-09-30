import os
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Historical Analysis",
    page_icon="📊",
)

st.title("📊 Historical Market Analysis")

st.write(
    "Explore historical Bitcoin price, return, volatility, " "momentum, and trend data."
)

BATCH_API = os.getenv("BATCH_API", "http://localhost:1062")

# -----------------------------------------------------------------------------
# Analysis inputs
# -----------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1: batch_symbol = st.selectbox( "Symbol", options=["BTCUSDT"], index=0, )

with col2: batch_interval = st.selectbox( "Interval", options=[ "1m", "5m", "15m", "30m", "1h", "4h", "1d", ], index=2, )

# -----------------------------------------------------------------------------
# Date range
# -----------------------------------------------------------------------------

date_col1, date_col2 = st.columns(2)

with date_col1: start_date = st.date_input( "Start date", value=datetime(2023, 1, 1).date(), )

with date_col2: end_date = st.date_input( "End date", value=datetime(2023, 1, 3).date(), )

# -----------------------------------------------------------------------------
# Analyze button
# -----------------------------------------------------------------------------

analyze_button = st.button( "📊 Analyze Historical Data", type="primary", use_container_width=True, )

# -----------------------------------------------------------------------------
# Run batch analysis
# -----------------------------------------------------------------------------

if analyze_button:

    if start_date > end_date:

        st.error(
            "❌ Start date must be before or equal to end date."
        )

    else:

        with st.spinner(
            "Fetching Binance data and running analysis..."
        ):

            try:

                response = requests.get(
                    f"{BATCH_API}/analysis",
                    params={
                        "symbol": batch_symbol,
                        "interval": batch_interval,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

                st.session_state["batch_result"] = result

                st.success("✅ Historical analysis completed.")

            except requests.exceptions.HTTPError:

                st.error(
                    f"❌ Batch API error "
                    f"{response.status_code}: "
                    f"{response.text}"
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Could not connect to Batch API: {e}"
                )

            except Exception as e:

                st.error(
                    f"❌ Analysis failed: {e}"
                )

# =============================================================================
# Display batch analysis results
# =============================================================================

if "batch_result" in st.session_state:

    result = st.session_state["batch_result"]

    market = result["market_summary"]

    st.subheader("📈 Market Summary")

    start_time = pd.to_datetime(market["start_time"])
    end_time = pd.to_datetime(market["end_time"])

    st.caption(
        f"**{market['symbol']}** · "
        f"{start_time.strftime('%d %b %Y')} → "
        f"{end_time.strftime('%d %b %Y')} · "
        f"{market['records']:,} candles"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Latest Price",
            f"${market['latest_price']:,.2f}",
        )

    with col2:
        st.metric(
            "Period High",
            f"${market['highest_price']:,.2f}",
        )

    with col3:
        st.metric(
            "Period Low",
            f"${market['lowest_price']:,.2f}",
        )

    with col4:
        st.metric(
            "Average Price",
            f"${market['average_price']:,.2f}",
        )

    st.divider()

    # =============================================================================
    # Returns and Volatility
    # =============================================================================

    returns = result["returns"]
    volatility = result["volatility"]

    col1, col2 = st.columns(2)

    # -----------------------------------------------------------------------------
    # Return Analysis
    # -----------------------------------------------------------------------------

    with col1:

        st.subheader("📈 Returns")

        st.metric(
            "Mean Return",
            f"{returns['mean']:.4%}",
        )

        st.metric(
            "Median Return",
            f"{returns['median']:.4%}",
        )

        st.metric(
            "Return Std",
            f"{returns['std']:.4%}",
        )

        st.metric(
            "Minimum",
            f"{returns['minimum']:.4%}",
        )

        st.metric(
            "Maximum",
            f"{returns['maximum']:.4%}",
        )


    # -----------------------------------------------------------------------------
    # Volatility Analysis
    # -----------------------------------------------------------------------------

    with col2:

        st.subheader("📉 Volatility")

        st.metric(
            "Mean",
            f"{volatility['mean']:.4%}",
        )

        st.metric(
            "Median",
            f"{volatility['median']:.4%}",
        )

        st.metric(
            "Latest",
            f"{volatility['latest']:.4%}",
        )

        st.metric(
            "Minimum",
            f"{volatility['minimum']:.4%}",
        )

        st.metric(
            "Maximum",
            f"{volatility['maximum']:.4%}",
        )

    st.divider()

    # =============================================================================
    # RSI and Trend
    # =============================================================================

    rsi = result["rsi"]
    trend = result["trend"]

    st.subheader("📊 Momentum & Trend")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------------------------------
    # RSI
    # -----------------------------------------------------------------------------

    with col1:

        st.markdown("### 📊 RSI")

        st.metric(
            "Latest RSI",
            f"{rsi['latest']:.2f}",
        )

        st.metric(
            "Average RSI",
            f"{rsi['average']:.2f}",
        )

        st.write(
            f"**RSI Range:** "
            f"{rsi['minimum']:.2f} – {rsi['maximum']:.2f}"
        )

        condition = rsi["condition"].capitalize()

        st.info(
            f"Current condition: **{condition}**"
        )


    # -----------------------------------------------------------------------------
    # Trend
    # -----------------------------------------------------------------------------

    with col2:

        st.markdown("### 📈 Trend")

        st.metric(
            "Current Price",
            f"${trend['price']:,.2f}",
        )

        st.metric(
            "EMA 20",
            f"${trend['ema_20']:,.2f}",
        )

        st.metric(
            "EMA 50",
            f"${trend['ema_50']:,.2f}",
        )

        st.write(
            f"**Distance from EMA 20:** "
            f"{trend['distance_from_ema_20']:.2%}"
        )

        st.write(
            f"**Above EMA 20:** "
            f"{'Yes' if trend['above_ema_20'] else 'No'}"
        )

        st.write(
            f"**Above EMA 50:** "
            f"{'Yes' if trend['above_ema_50'] else 'No'}"
        )

    st.divider()

    # =============================================================================
    # Historical Price Chart
    # =============================================================================

    price_series = result["price_series"]

    st.subheader("📊 Price History")

    if price_series:

        chart_df = pd.DataFrame(price_series)

        chart_df["time"] = pd.to_datetime(
            chart_df["time"]
        )

        fig = go.Figure()

        # -------------------------------------------------------------------------
        # Closing Price
        # -------------------------------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=chart_df["time"],
                y=chart_df["close"],
                mode="lines",
                name="Close",
            )
        )

        # -------------------------------------------------------------------------
        # EMA 20
        # -------------------------------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=chart_df["time"],
                y=chart_df["ema_20"],
                mode="lines",
                name="EMA 20",
            )
        )

        # -------------------------------------------------------------------------
        # EMA 50
        # -------------------------------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=chart_df["time"],
                y=chart_df["ema_50"],
                mode="lines",
                name="EMA 50",
            )
        )

        fig.update_layout(
            height=500,
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            hovermode="x unified",
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        fig.update_xaxes(
            rangeslider_visible=True,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.warning(
            "No price series data is available for this analysis."
        )