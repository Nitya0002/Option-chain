# streamlit_option_liquidity_simulation.py
# ---------------------------------------------------------
# Auto-installs required packages if missing (for GitHub/Cloud use)
# ---------------------------------------------------------

import importlib
import subprocess
import sys

required = ["streamlit", "matplotlib", "pandas"]

for pkg in required:
    if importlib.util.find_spec(pkg) is None:
        print(f"Installing missing package: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Now safe to import
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# App logic starts here
# ---------------------------------------------------------
st.set_page_config(page_title="Illiquid Option Simulation", layout="wide")

st.title("Illiquid Option & Orderbook Simulation — Educational")
st.markdown(
    """
    This Streamlit app simulates a small illiquid option orderbook.
    It’s **educational**, showing how wide spreads and sparse liquidity can cause
    a human trader’s order to move prices or incur losses.

    ⚠️ *Not for trading use.*
    """
)

# Sidebar parameters
with st.sidebar.form("params"):
    st.header("Simulation parameters")
    fair_price = st.number_input("Fair price", value=40.0, step=1.0)
    initial_bid = st.number_input("Initial bid", value=20.0, step=1.0)
    initial_ask = st.number_input("Initial ask", value=80.0, step=1.0)
    initial_size = st.number_input("Bid/Ask size", value=100, step=10)
    mm_spread = st.number_input("Market maker spread", value=5.0, step=0.5)
    mm_size = st.number_input("Market maker size", value=50, step=5)
    volatility = st.number_input("Volatility (noise)", value=2.5, step=0.1)
    steps = st.slider("Simulation steps", 10, 200, 30)
    st.markdown("---")
    human_limit_price = st.number_input("Human buy price", value=21.0, step=0.5)
    human_size = st.number_input("Human buy size", value=10, step=1)
    run_btn = st.form_submit_button("Run Simulation")

if not run_btn:
    st.stop()

# --------------------- Simulation ---------------------

book = {"bids": [(initial_bid, initial_size)], "asks": [(initial_ask, initial_size)]}

def mm_quotes(fair, spread, size):
    return (round(fair - spread, 2), size), (round(fair + spread, 2), size)

def get_mid(b):
    if not b["bids"] or not b["asks"]:
        return None
    return (b["bids"][0][0] + b["asks"][0][0]) / 2

# add MM
mm_bid, mm_ask = mm_quotes(fair_price, mm_spread, mm_size)
book["bids"].append(mm_bid)
book["asks"].append(mm_ask)
book["bids"].sort(key=lambda x: -x[0])
book["asks"].sort(key=lambda x: x[0])

history, trades = [], []
human_filled, human_avg_price = 0, None

for t in range(steps):
    mid = get_mid(book) or fair_price
    drift = (fair_price - mid) * 0.05
    shock = random.gauss(0, volatility)
    _indic = mid + drift + shock * 0.1

    # Random taker
    if random.random() < 0.25:
        if random.random() < 0.5 and book["bids"]:
            p, s = book["bids"][0]
            take = min(s, random.randint(1, s))
            trades.append({"time": t, "price": p, "side": "sell_into_bid"})
            if s - take <= 0:
                book["bids"].pop(0)
            else:
                book["bids"][0] = (p, s - take)
        elif book["asks"]:
            p, s = book["asks"][0]
            take = min(s, random.randint(1, s))
            trades.append({"time": t, "price": p, "side": "buy_from_ask"})
            if s - take <= 0:
                book["asks"].pop(0)
            else:
                book["asks"][0] = (p, s - take)

    mm_bid, mm_ask = mm_quotes(fair_price + random.gauss(0, 1.0), mm_spread, mm_size)
    if mm_bid not in book["bids"]:
        book["bids"].append(mm_bid)
    if mm_ask not in book["asks"]:
        book["asks"].append(mm_ask)
    book["bids"].sort(key=lambda x: -x[0])
    book["asks"].sort(key=lambda x: x[0])

    # Human order fill
    if human_filled < human_size and book["asks"] and book["asks"][0][0] <= human_limit_price:
        ask_price, ask_size = book["asks"][0]
        fill_qty = min(human_size - human_filled, ask_size)
        trades.append({"time": t, "price": ask_price, "side": "human_buy_fill"})
        if human_avg_price is None:
            human_avg_price = ask_price
        else:
            human_avg_price = (human_avg_price * human_filled + ask_price * fill_qty) / (human_filled + fill_qty)
        human_filled += fill_qty
        if ask_size - fill_qty <= 0:
            book["asks"].pop(0)
        else:
            book["asks"][0] = (ask_price, ask_size - fill_qty)

    history.append({
        "time": t,
        "bid": book["bids"][0][0] if book["bids"] else None,
        "ask": book["asks"][0][0] if book["asks"] else None,
        "mid": get_mid(book)
    })

# --------------------- Results ---------------------

df_hist = pd.DataFrame(history)
df_trades = pd.DataFrame(trades)

st.subheader("Price Evolution")
fig, ax = plt.subplots()
ax.plot(df_hist["time"], df_hist["mid"], marker="o", label="Mid price")
ax.set_xlabel("Step")
ax.set_ylabel("Price")
ax.grid(True)
st.pyplot(fig)

st.subheader("Trade Events")
st.dataframe(df_trades)

st.markdown("---")
if human_filled:
    pnl = (fair_price - human_avg_price) * human_filled
    st.success(f"Filled {human_filled} @ {human_avg_price:.2f} | P&L vs fair: {pnl:.2f}")
else:
    st.warning("Human order did not fill.")
