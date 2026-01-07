import random
import heapq
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplfinance as mpf

# =====================================================
# GLOBAL SETTINGS
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_ORDERS = 1000
START_PRICE = 100.0

# =====================================================
# ORDER & ORDER BOOK
# =====================================================
class Order:
    def __init__(self, oid, side, price, qty, timestamp):
        self.id = oid
        self.side = side
        self.price = price
        self.qty = qty
        self.remaining = qty
        self.timestamp = timestamp

class OrderBook:
    def __init__(self):
        self.bids = []   # (-price, seq, order)
        self.asks = []   # ( price, seq, order)
        self.seq = itertools.count()
        self.trades = []

    def add_order(self, order):
        if order.side == "BUY":
            self._match_buy(order)
            if order.remaining > 0:
                heapq.heappush(self.bids, (-order.price, next(self.seq), order))
        else:
            self._match_sell(order)
            if order.remaining > 0:
                heapq.heappush(self.asks, (order.price, next(self.seq), order))

    def _match_buy(self, buy):
        while buy.remaining > 0 and self.asks:
            ask_price, _, ask = self.asks[0]
            if buy.price < ask_price:
                break
            qty = min(buy.remaining, ask.remaining)
            self._record_trade(ask_price, qty, buy.id, ask.id)
            buy.remaining -= qty
            ask.remaining -= qty
            if ask.remaining == 0:
                heapq.heappop(self.asks)

    def _match_sell(self, sell):
        while sell.remaining > 0 and self.bids:
            bid_price, _, bid = self.bids[0]
            bid_price = -bid_price
            if sell.price > bid_price:
                break
            qty = min(sell.remaining, bid.remaining)
            self._record_trade(bid_price, qty, bid.id, sell.id)
            sell.remaining -= qty
            bid.remaining -= qty
            if bid.remaining == 0:
                heapq.heappop(self.bids)

    def _record_trade(self, price, qty, buyer, seller):
        assert price >= 0 and qty > 0
        self.trades.append({
            "timestamp": current_time,
            "price": price,
            "quantity": qty,
            "buyer": buyer,
            "seller": seller
        })

# =====================================================
# SIMULATION: 1,000 RANDOM ORDERS
# =====================================================
book = OrderBook()
current_time = 0.0
price = START_PRICE

for i in range(NUM_ORDERS):
    current_time += random.expovariate(1 / 2)
    side = "BUY" if random.random() < 0.5 else "SELL"
    price += np.random.normal(0, 0.2)
    price = max(price, 1.0)
    qty = random.randint(1, 10)

    order = Order(
        oid=f"O{i}",
        side=side,
        price=round(price, 2),
        qty=qty,
        timestamp=current_time
    )

    book.add_order(order)

# =====================================================
# TRADE TAPE & OHLC
# =====================================================
trades_df = pd.DataFrame(book.trades)
assert not trades_df.empty, "No trades executed"

trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], unit="s")
trades_df.set_index("timestamp", inplace=True)

ohlc = trades_df["price"].resample("1min").ohlc().dropna()

# =====================================================
# CANDLESTICK STYLE (IMPROVED COLORS)
# =====================================================
market_colors = mpf.make_marketcolors(
    up="green",
    down="red",
    edge="inherit",
    wick="inherit",
    volume="inherit"
)

style = mpf.make_mpf_style(
    marketcolors=market_colors,
    gridstyle="--",
    gridcolor="lightgray",
    facecolor="white",
    edgecolor="black",
    rc={
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14
    }
)

# =====================================================
# PDF REPORT
# =====================================================
with PdfPages("simulation_report.pdf") as pdf:

    # Page 1 — Setup
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.text(0.1, 0.8, "Simulation Report", fontsize=16, weight="bold")
    ax.text(0.1, 0.6, f"Random Orders: {NUM_ORDERS}")
    ax.text(0.1, 0.5, f"Seed: {SEED}")
    ax.text(0.1, 0.4, "Candlestick derived from Trade Tape (1-min OHLC)")
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2 — Candlestick Chart
    fig, _ = mpf.plot(
        ohlc,
        type="candle",
        style=style,
        title="Price Candlestick (1-Minute OHLC)",
        ylabel="Price",
        volume=False,
        returnfig=True
    )
    pdf.savefig(fig)
    plt.close(fig)

print("simulation_report.pdf generated successfully")
