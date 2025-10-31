"""
Streamlit app: Illiquid Option / Orderbook Simulation (educational)

This app simulates an extremely simplified order book for an illiquid option to
illustrate how wide spreads and sparse liquidity can cause a human trader to
experience large slippage or loss. This is purely educational and NOT trading
advice. The simulation intentionally keeps mechanics simple so you can run it
locally and push to GitHub.

How to run:
1. Install requirements: pip install streamlit pandas matplotlib
2. Run: streamlit run streamlit_option_liquidity_simulation.py

Notes:
- The app models top-of-book only (one best bid and ask, plus a small market-maker
  posting near the fair price).
- A "human" limit buy can be entered; the simulation will show whether it filled
  and at what price. The benign MM re-posts quotes each step.
- The app intentionally avoids any realistic exchange API usage or order routing.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from io import StringIO

st.set_page_config(page_title="Illiquid Option Simulation", layout="wide")

st.title("Illiquid Option & Orderbook Simulation — Educational")
st.markdown(
    """
    This small Streamlit app simulates a tiny orderbook for an illiquid option.
    Use the controls to change the fair price, quotes, and the human order.
    The simulation demonstrates how an isolated human order can get filled (or
    not) and how mid-price / trade events evolve.

    **For educational purposes only.**
    """
)

# --- Controls ---
with st.sidebar.form(key="params"):
    st.header("Simulation parameters")
    fair_price = st.number_input("Fair price (reference)", value=40.0, step=1.0)
    initial_bid = st.number_input("Initial lone bid", value=20.0, step=1.0)
    initial_ask = st.number_input("Initial lone ask", value=80.0, step=1.0)
    initial_size = st.number_input("Size at lone quotes", value=100, step=1)

    mm_spread = st.number_input("MM spread around fair price", value=5.0, step=0.5)
    mm_size = st.number_input("MM size", value=50, step=1)

    volatility = st.number_input("Volatility (noise)", value=2.5, step=0.1)
    steps = st.slider("Simulation steps", min_value=5, max_value=200, value=30)

    st.write("---")
    st.header("Human order")
    human_limit_price = st.number_input("Human limit buy price", value=21.0, step=0.5)
    human_size = st.number_input("Human order size", value=10, step=1)
    run_button = st.form_submit_button("Run simulation")

if not run_button:
    st.info("Change parameters on the left and click 'Run simulation' to start.")
    st.stop()

# --- Simulation code ---
# Basic order-book: top-of-book only (list of one or two levels per side)
book = {
    "bids": [(round(initial_bid, 2), int(initial_size))],
    "asks": [(round(initial_ask, 2), int(initial_size))]
}

# helper: market-maker quotes
def mm_quotes(fair, spread, size):
    return (round(max(0.01, fair - spread), 2), size), (round(fair + spread, 2), size)

# helper: mid price
def get_mid(book):
    if not book["bids"] or not book["asks"]:
        return None
    return (book["bids"][0][0] + book["asks"][0][0]) / 2.0

# Add benign MM initially
mm_bid, mm_ask = mm_quotes(fair_price, mm_spread, mm_size)
# Avoid duplicate identical tuple copies
if mm_bid not in book["bids"]:
    book["bids"].append(mm_bid)
if mm_ask not in book["asks"]:
    book["asks"].append(mm_ask)
# sort
book["bids"] = sorted(book["bids"], key=lambda x: -x[0])
book["asks"] = sorted(book["asks"], key=lambda x: x[0])

# Records
history = []
trades = []

human_filled = 0
human_avg_price = None

for t in range(steps):
    # simple drift toward fair price plus gaussian shock
    mid = get_mid(book)
    if mid is None:
        mid = fair_price
    # small pull toward fair
    drift = (fair_price - mid) * 0.05
    shock = random.gauss(0, volatility)
    _indicative = mid + drift + shock * 0.1

    # Random liquidity taker consumes top-of-book
    if random.random() < 0.25:
        # randomly remove from bid or ask
        if random.random() < 0.5 and book["bids"]:
            p, s = book["bids"][0]
            take = min(s, random.randint(1, max(1, int(s))))
            trades.append({"time": t, "price": p, "size": take, "side": "sell_into_bid"})
            if s - take <= 0:
                book["bids"].pop(0)
            else:
                book["bids"][0] = (p, s - take)
        elif book["asks"]:
            p, s = book["asks"][0]
            take = min(s, random.randint(1, max(1, int(s))))
            trades.append({"time": t, "price": p, "size": take, "side": "buy_from_ask"})
            if s - take <= 0:
                book["asks"].pop(0)
            else:
                book["asks"][0] = (p, s - take)

    # Re-post MM near fair each step (simple refresh)
    mm_bid, mm_ask = mm_quotes(fair_price + random.gauss(0, 1.0), mm_spread, mm_size)
    if mm_bid not in book["bids"]:
        book["bids"].append(mm_bid)
    if mm_ask not in book["asks"]:
        book["asks"].append(mm_ask)
    book["bids"] = sorted(book["bids"], key=lambda x: -x[0])
    book["asks"] = sorted(book["asks"], key=lambda x: x[0])

    # Human limit order: fill if best ask <= human_limit_price
    if human_filled < human_size and book["asks"] and book["asks"][0][0] <= human_limit_price:
        ask_price, ask_size = book["asks"][0]
        fill_qty = min(human_size - human_filled, ask_size)
        trades.append({"time": t, "price": ask_price, "size": fill_qty, "side": "human_buy_fill"})
        if human_avg_price is None:
            human_avg_price = ask_price
        else:
            human_avg_price = (human_avg_price * human_filled + ask_price * fill_qty) / (human_filled + fill_qty)
        human_filled += fill_qty
        # reduce or remove
        if ask_size - fill_qty <= 0:
            book["asks"].pop(0)
        else:
            book["asks"][0] = (ask_price, ask_size - fill_qty)

    # record
    snapshot = {
        "time": t,
        "best_bid": book["bids"][0][0] if book["bids"] else None,
        "best_bid_size": book["bids"][0][1] if book["bids"] else None,
        "best_ask": book["asks"][0][0] if book["asks"] else None,
        "best_ask_size": book["asks"][0][1] if book["asks"] else None,
        "mid": get_mid(book)
    }
    history.append(snapshot)

    # break if liquidity disappears
    if not book["bids"] or not book["asks"]:
        break

# compute human pnl vs fair
human_pnl = None
if human_filled > 0:
    human_pnl = (fair_price - human_avg_price) * human_filled

# Convert to DataFrames
df_history = pd.DataFrame(history)
df_trades = pd.DataFrame(trades)

# --- UI output ---
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Top-of-book mid-price evolution")
    fig, ax = plt.subplots(figsize=(8, 3))
    if not df_history.empty:
        ax.plot(df_history["time"], df_history["mid"], marker='o')
    if not df_trades.empty:
        buys = df_trades[df_trades["side"].str.contains("buy")]
        sells = df_trades[df_trades["side"].str.contains("sell")]
        if not buys.empty:
            ax.scatter(buys["time"], buys["price"], marker='^', label='buys')
        if not sells.empty:
            ax.scatter(sells["time"], sells["price"], marker='v', label='sells')
    ax.set_xlabel("time step")
    ax.set_ylabel("price")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Trade events")
    st.dataframe(df_trades.reset_index(drop=True))

with col2:
    st.subheader("Simulation summary")
    st.write(f"Fair price (reference): **{fair_price}**")
    st.write(f"Initial top bid/ask: **{initial_bid} / {initial_ask}** (size {initial_size})")
    st.write(f"MM spread ±{mm_spread} with size {mm_size}")
    st.write(f"Human limit buy @ **{human_limit_price}** for size **{human_size}**")
    st.write(f"Human executed quantity: **{human_filled}/{human_size}**")
    if human_filled > 0:
        st.write(f"Human average entry price: **{human_avg_price:.2f}**")
        st.write(f"Human P&L vs fair price: **{human_pnl:.2f}** (positive means profit)")
    else:
        st.write("Human order did not fill during the simulation window.")

# Download CSVs
if not df_history.empty:
    csv_hist = df_history.to_csv(index=False)
    st.download_button("Download orderbook snapshots (CSV)", data=csv_hist, file_name="orderbook_snapshots.csv")
if not df_trades.empty:
    csv_trades = df_trades.to_csv(index=False)
    st.download_button("Download trade events (CSV)", data=csv_trades, file_name="trade_events.csv")

st.markdown("---")
st.caption("Educational simulation only — does not represent real exchange behavior.")
