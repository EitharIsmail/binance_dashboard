"""
Streamlit dashboard for the Binance Direction Classifier API.

No user controls for now -- horizon is fixed to 30m and symbol is fixed
to BTCUSDT. ONLINE_API points at the Docker-internal service name
(http://api:8000) because this script's requests.post() calls run
server-side in the Streamlit Python process, not in the user's browser --
unlike a client-side JS fetch, there's no need to route this call through
the public nginx domain.
"""
import os
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st

ONLINE_API = os.getenv("ONLINE_API", "http://localhost:8000")

FIXED_HORIZON = "30m"
FIXED_HORIZON_MINUTES = 30
FIXED_SYMBOL_DISPLAY = "Bitcoin (BTC/USDT)"

st.set_page_config(page_title="Bitcoin Direction Predictor", page_icon="₿", layout="centered")

st.title("₿ Bitcoin Direction Predictor")
st.markdown(
    f"This tool predicts whether **Bitcoin's price** will go significantly "
    f"**up**, significantly **down**, or stay **roughly unchanged**, over "
    f"the **next {FIXED_HORIZON_MINUTES} minutes** from right now."
)
st.caption(f"Currently supports Bitcoin only, on a fixed {FIXED_HORIZON} horizon.")


@st.cache_data(ttl=15)
def get_health():
    try:
        r = requests.get(f"{ONLINE_API}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


health = get_health()
loaded_horizons = health.get("loaded_horizons", {}) if health else {}
horizon_ready = FIXED_HORIZON in loaded_horizons

if health and health.get("status") == "ok" and horizon_ready:
    meta = loaded_horizons[FIXED_HORIZON]
    st.success(
        f"✅ System ready — serving **{meta.get('model_name', '?')}** "
        f"v{meta.get('version', '?')} @{meta.get('model_alias', '?')}"
    )
elif health and health.get("status") == "ok":
    st.error(f"⚠️ The API is running, but no {FIXED_HORIZON} Bitcoin model is currently available.")
else:
    st.error(f"❌ Cannot reach the prediction service at `{ONLINE_API}`. Please try again shortly.")

st.divider()

st.header("🔮 Get a Live Prediction")
st.write(
    f"Click below to fetch the latest Bitcoin price data and predict "
    f"where the price is headed over the next {FIXED_HORIZON_MINUTES} minutes."
)

if st.button("▶️ Predict Bitcoin's next 30 minutes", type="primary", disabled=not horizon_ready):
    request_time = datetime.now(timezone.utc)
    target_time = request_time + timedelta(minutes=FIXED_HORIZON_MINUTES)

    try:
        r = requests.post(f"{ONLINE_API}/predict", json={"horizon": FIXED_HORIZON}, timeout=15)
        if r.status_code == 200:
            result = r.json()
            direction = result["prediction"]

            direction_labels = {
                "Bull": ("🟢", "Price is likely to go UP"),
                "Bear": ("🔴", "Price is likely to go DOWN"),
                "Neutral": ("🟡", "Price is likely to stay roughly UNCHANGED"),
            }
            emoji, description = direction_labels.get(direction, ("⚪", direction))

            st.metric(f"{emoji} Prediction: {direction}", description)
            st.info(
                f"📅 **Prediction made at:** {request_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
                f"🎯 **This forecast is for:** {target_time.strftime('%Y-%m-%d %H:%M:%S')} UTC "
                f"(i.e. {FIXED_HORIZON_MINUTES} minutes from now)\n\n"
                f"💰 **Asset:** {FIXED_SYMBOL_DISPLAY}"
            )
            st.caption(f"Model used: {result['model_name']} v{result['model_version']}")
        else:
            st.error(f"API error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.divider()

with st.expander("ℹ️ How this works"):
    st.write(
        "This dashboard fetches Bitcoin's most recent price and trading "
        "data from Binance, computes technical indicators (momentum, "
        "volatility, moving averages, RSI, volume), and feeds them into "
        "a trained machine learning model. The model was trained "
        "specifically for a 30-minute horizon on Bitcoin -- it does not "
        "currently support other cryptocurrencies or other time horizons."
    )

if st.button("🔄 Refresh model status"):
    st.cache_data.clear()
    st.rerun()