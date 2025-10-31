# streamlit_option_liquidity_simulation_enhanced.py
"""
Enhanced Illiquid Option / Orderbook Simulation (Streamlit)

Features:
- Clean two-column layout with controls on the left.
- Metrics for fair price, current mid, spread, human P&L.
- Line chart (matplotlib) of bid/ask/mid + trade markers.
- Trade events table and histogram of trade prices.
- Live simulation mode with adjustable speed, or run-to-complete mode.
- CSV download for snapshots and trades.
- Educational summary text.

Run:
streamlit run streamlit_option_liquidity_simulation_enhanced.py
"""
from typing import List, Tuple, Dict, Any
import random
import time
import io

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

st.set_page_config(page_title="Illiquid Option Simulator", layout="wide")

# ---- Helper functions ----
def mm_quotes(fair: float, spread: float, size: int) -> Tuple[Tuple[float,int], Tuple[float,int]]:
    return (round(max(0.01, fair - spread), 2), size), (round(fair + spread, 2), size)

def get_mid(book):
    if not book["bids"] or not book["asks"]:
        return None
    return (book["bids"][0][0] + book["asks"][0][0]) / 2.0

def snapshot_to_df(history: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(history)

def trades_to_df(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(trades)

def compute_pnl(fair, entry_price, qty):
    return (fair - entry_price) * qty

def download_csv_bytes(df: pd.DataFrame, name: str) -> Tuple[bytes, str]:
    csv = df.to_csv(index=False)
    return csv.encode('utf-8'), f"{name}.csv"

# ---- UI: controls ----
st.title("📊 Illiquid Option — Enhanced Simulator")
st.markdown(
    "Educational demo showing how wide spreads and sparse liquidity can cause slippage. "
    "Not trading advice."
)

with st.sidebar:
    st.header("Simulation controls")
    fair_price = st.number_input("Fair price (reference)", value=40.0, step=0.5)
    initial_bid = st.number_input("Initial lone bid", value=20.0, step=0.5)
    initial_ask = st.number_input("Initial lone ask", value=80.0, step=0.5)
    initial_size = st.number_input("Size at lone quotes", value=100, step=1)
    st.write("---")
    mm_spread = st.number_input("Market-maker spread (±)", value=5.0, step=0.5)
    mm_size = st.number_input("MM size", value=50, step=1)
    volatility = st.number_input("Volatility (noise σ)", value=2.5, step=0.1)
    steps = st.slider("Simulation steps (max)", 5, 500, 60)
    live_mode = st.checkbox("Live simulation (animate steps)", value=False)
    speed = st.slider("Live speed (seconds per step)", 0.05, 1.0, 0.15, step=0.05)
    st.write("---")
    st.header("Human order")
    human_limit_price = st.number_input("Human limit buy price", value=21.0, step=0.5)
    human_size = st.number_input("Human order size", value=10, step=1)
    st.write("---")
    actions = st.columns([1,1])
    run_button = actions[0].button("▶ Run")
    reset_button = actions[1].button("⟲ Reset")

# ---- Simulation state reset when Reset clicked ----
if reset_button:
    st.experimental_rerun()

# ---- Prepare initial orderbook ----
book = {
    "bids": [(round(initial_bid,2), int(initial_size))],
    "asks": [(round(initial_ask,2), int(initial_size))]
}
# Add MM
mm_bid, mm_ask = mm_quotes(fair_price, mm_spread, mm_size)
if mm_bid not in book["bids"]:
    book["bids"].append(mm_bid)
if mm_ask not in book["asks"]:
    book["asks"].append(mm_ask)
book["bids"] = sorted(book["bids"], key=lambda x: -x[0])
book["asks"] = sorted(book["asks"], key=lambda x: x[0])

# ---- Data containers ----
history = []
trades = []
human_filled = 0
human_avg_price = None

# Utility to render charts and tables inside main loop
def render_ui(history, trades, human_filled, human_avg_price):
    df_hist = snapshot_to_df(history) if history else pd.DataFrame()
    df_trades = trades_to_df(trades) if trades else pd.DataFrame()

    col_main, col_side = st.columns([3,1])

    # Left: Chart + trades
    with col_main:
        st.subheader("Price chart — bid / ask / mid")
        fig, ax = plt.subplots(figsize=(9,4))
        if not df_hist.empty:
            x = df_hist["time"]
            # plot bid, ask
            ax.plot(x, df_hist["best_bid"], linestyle='-', marker='o', label='best bid')
            ax.plot(x, df_hist["best_ask"], linestyle='-', marker='o', label='best ask')
            ax.plot(x, df_hist["mid"], linestyle='--', linewidth=1.5, label='mid')
            # shade fair price horizontally
            ax.axhline(fair_price, color='gray', linestyle=':', label='fair price')
        # plot trades
        if not df_trades.empty:
            buys = df_trades[df_trades["side"].str.contains("buy", na=False)]
            sells = df_trades[df_trades["side"].str.contains("sell", na=False)]
            if not buys.empty:
                ax.scatter(buys["time"], buys["price"], marker='^', s=80, label='buys', zorder=5)
            if not sells.empty:
                ax.scatter(sells["time"], sells["price"], marker='v', s=80, label='sells', zorder=5)
        ax.set_xlabel("step")
        ax.set_ylabel("price")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        st.pyplot(fig)

        st.markdown("### Trade events")
        st.dataframe(df_trades.reset_index(drop=True))

        # Histogram of trade prices
        if not df_trades.empty:
            st.markdown("### Trade price distribution")
            fig2, ax2 = plt.subplots(figsize=(6,2.5))
            ax2.hist(df_trades["price"], bins=12)
            ax2.set_xlabel("trade price")
            ax2.set_ylabel("count")
            st.pyplot(fig2)

    # Right: summary metrics and CSV downloads
    with col_side:
        st.subheader("Summary")
        mid_latest = df_hist["mid"].iloc[-1] if (not df_hist.empty) else None
        spread_latest = None
        if mid_latest is not None:
            spread_latest = (df_hist["best_ask"].iloc[-1] - df_hist["best_bid"].iloc[-1])
        # Show metrics
        st.metric("Fair price", f"{fair_price:.2f}")
        if mid_latest is not None:
            st.metric("Current mid", f"{mid_latest:.2f}", delta=f"{(mid_latest - fair_price):+.2f}")
            st.write(f"Latest spread: **{spread_latest:.2f}**" if spread_latest is not None else "Spread: N/A")
        else:
            st.write("No market snapshot yet.")

        st.write("---")
        st.write("Human order")
        st.write(f"Limit buy: **{human_limit_price}**  | Size: **{human_size}**")
        if human_filled > 0:
            pnl = compute_pnl(fair_price, human_avg_price, human_filled)
            col_a, col_b = st.columns([1,1])
            col_a.metric("Filled qty", f"{human_filled}/{human_size}")
            col_b.metric("Avg entry", f"{human_avg_price:.2f}")
            if pnl >= 0:
                st.success(f"Human P&L vs fair: {pnl:.2f}")
            else:
                st.error(f"Human P&L vs fair: {pnl:.2f}")
        else:
            st.info("Human order not filled yet.")

        st.write("---")
        # Download CSVs
        if df_hist.shape[0] > 0:
            hist_bytes, hist_name = download_csv_bytes(df_hist, "snapshots")
            st.download_button("⬇ Download snapshots CSV", data=hist_bytes, file_name=hist_name, mime="text/csv")
        if df_trades.shape[0] > 0:
            trades_bytes, trades_name = download_csv_bytes(df_trades, "trades")
            st.download_button("⬇ Download trades CSV", data=trades_bytes, file_name=trades_name, mime="text/csv")

        st.write("---")
        st.caption("Educational simulation — not real trading.")

# ---- Main simulation runner ----
if run_button:
    # We'll run up to `steps`. If live_mode is set, we animate with small sleep.
    for t in range(steps):
        # Natural drift toward fair + noise
        mid = get_mid(book) or fair_price
        drift = (fair_price - mid) * 0.05
        shock = random.gauss(0, volatility)
        _indic = mid + drift + shock * 0.1

        # Random liquidity taker event
        if random.random() < 0.25:
            if random.random() < 0.5 and book["bids"]:
                p, s = book["bids"][0]
                take = min(s, random.randint(1, max(1, int(s*0.2))))
                trades.append({"time": t, "price": p, "size": take, "side": "sell_into_bid"})
                if s - take <= 0:
                    book["bids"].pop(0)
                else:
                    book["bids"][0] = (p, s - take)
            elif book["asks"]:
                p, s = book["asks"][0]
                take = min(s, random.randint(1, max(1, int(s*0.2))))
                trades.append({"time": t, "price": p, "size": take, "side": "buy_from_ask"})
                if s - take <= 0:
                    book["asks"].pop(0)
                else:
                    book["asks"][0] = (p, s - take)

        # MM refresh quotes
        mm_bid, mm_ask = mm_quotes(fair_price + random.gauss(0,1.0), mm_spread, mm_size)
        if mm_bid not in book["bids"]:
            book["bids"].append(mm_bid)
        if mm_ask not in book["asks"]:
            book["asks"].append(mm_ask)
        book["bids"] = sorted(book["bids"], key=lambda x: -x[0])
        book["asks"] = sorted(book["asks"], key=lambda x: x[0])

        # Human order fill logic
        if human_filled < human_size and book["asks"] and book["asks"][0][0] <= human_limit_price:
            ask_price, ask_size = book["asks"][0]
            fill_qty = min(human_size - human_filled, ask_size)
            trades.append({"time": t, "price": ask_price, "size": fill_qty, "side": "human_buy_fill"})
            if human_avg_price is None:
                human_avg_price = ask_price
            else:
                human_avg_price = (human_avg_price * human_filled + ask_price * fill_qty) / (human_filled + fill_qty)
            human_filled += fill_qty
            # reduce / remove ask
            if ask_size - fill_qty <= 0:
                book["asks"].pop(0)
            else:
                book["asks"][0] = (ask_price, ask_size - fill_qty)

        # record snapshot
        snapshot = {
            "time": t,
            "best_bid": book["bids"][0][0] if book["bids"] else None,
            "best_bid_size": book["bids"][0][1] if book["bids"] else None,
            "best_ask": book["asks"][0][0] if book["asks"] else None,
            "best_ask_size": book["asks"][0][1] if book["asks"] else None,
            "mid": get_mid(book)
        }
        history.append(snapshot)

        # Update UI progressively in live mode
        if live_mode:
            render_ui(history, trades, human_filled, human_avg_price)
            time.sleep(speed)
            # clear for next frame (so chart updates instead of stacking many charts)
            st.experimental_rerun()  # cause a rerun to update state and UI (keeps code simple)

        # Stop early if liquidity collapses
        if not book["bids"] or not book["asks"]:
            break

    # finished running (non-live or live didn't abort)
    render_ui(history, trades, human_filled, human_avg_price)

    # Text summary explanation
    st.markdown("## What happened?")
    if human_filled == 0:
        st.info(
            "The human limit buy did not execute within the simulation steps. "
            "In an illiquid market the spread can persist and the order may rest unfilled."
        )
    else:
        pnl = compute_pnl(fair_price, human_avg_price, human_filled)
        if pnl < 0:
            st.error(
                f"The human bought above fair price and would have an unrealized loss of {pnl:.2f} "
                "versus the fair value. This demonstrates slippage in illiquid markets."
            )
        else:
            st.success(
                f"The human achieved a positive P&L ({pnl:.2f}) vs fair price."
            )

else:
    st.info("Configure parameters and press ▶ Run to start the simulation.")

# ---- Notes and download hints ----
st.markdown("---")
st.markdown(
    "**Notes:**\n\n- This is a simplified, deterministic simulation for teaching market microstructure.\n"
    "- For production-quality orderbook simulation you'd use many price levels, latency, and a matching engine.\n"
)
