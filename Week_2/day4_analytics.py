import pandas as pd
import numpy as np

# =====================================================
# DAY 4 — INPUT DATA (TAPE + L1 SNAPSHOTS)
# =====================================================
# In real usage, these come from Day 3.
# Here we hardcode sample data ONLY to demonstrate Day 4.

trade_log = [
    {"timestamp": 4.0, "price": 101, "quantity": 10, "buyer": "B1", "seller": "A1"},
    {"timestamp": 4.0, "price": 102, "quantity": 20, "buyer": "B1", "seller": "A2"},
    {"timestamp": 4.0, "price": 103, "quantity": 30, "buyer": "B1", "seller": "A3"},
]

l1_snapshots = [
    {"timestamp": 0, "best_bid": None, "best_ask": None, "spread": None, "mid_price": None},
    {"timestamp": 1, "best_bid": None, "best_ask": 101, "spread": None, "mid_price": None},
    {"timestamp": 2, "best_bid": None, "best_ask": 101, "spread": None, "mid_price": None},
    {"timestamp": 3, "best_bid": None, "best_ask": 101, "spread": None, "mid_price": None},
    {"timestamp": 4, "best_bid": 101, "best_ask": 103, "spread": 2, "mid_price": 102},
    {"timestamp": 5, "best_bid": None, "best_ask": None, "spread": None, "mid_price": None},
]

# =====================================================
# DAY 4 — ANALYTICS PIPELINE
# =====================================================

# Convert to DataFrames
trades_df = pd.DataFrame(trade_log)
l1_df = pd.DataFrame(l1_snapshots)

# -------------------------
# VWAP (FROM TRADE TAPE)
# -------------------------
vwap = (
    trades_df["price"] * trades_df["quantity"]
).sum() / trades_df["quantity"].sum()

# -------------------------
# SPREAD (FROM L1 SNAPSHOTS)
# -------------------------
avg_spread = l1_df["spread"].dropna().mean()

# -------------------------
# MID-PRICE VOLATILITY
# -------------------------
mid_prices = l1_df["mid_price"].dropna()
log_returns = np.log(mid_prices).diff().dropna()
mid_price_volatility = log_returns.std()

# =====================================================
# OUTPUT
# =====================================================

print("\n===== DAY 4 ANALYTICS =====")
print(f"VWAP: {vwap:.4f}")
print(f"Average Spread: {avg_spread:.4f}")
print(f"Mid-Price Volatility: {mid_price_volatility:.6f}")

# =====================================================
# VALIDATION CHECKS (NON-NEGOTIABLE)
# =====================================================

assert trades_df["price"].min() <= vwap <= trades_df["price"].max()
assert (l1_df["spread"].dropna() >= 0).all()
assert mid_price_volatility >= 0

print("\n✅ Day 4 ONLY analytics verified successfully")
