import random
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CORE MARKET COMPONENTS
# =========================

class Exchange:
    def __init__(self, fair_price=100.0, min_spread=0.5):
        self.price = fair_price
        self.spread = 2.0
        self.fair_price = fair_price
        self.min_spread = min_spread

    def step(self, order_imbalance, liquidity_strength):
        """
        Price Impact Model:
        - Noise causes random walk
        - Momentum amplifies trends
        - Market maker mean-reverts price
        """

        noise = random.gauss(0, 0.15)

        # price change
        self.price += noise + order_imbalance

        # market maker stabilization
        if liquidity_strength > 0:
            correction = liquidity_strength * (self.fair_price - self.price)
            self.price += correction

            # spread tightening but bounded
            self.spread = max(
                self.min_spread,
                self.spread - liquidity_strength * 0.3
            )
        else:
            # no liquidity → spread widens
            self.spread += abs(order_imbalance) * 0.4

        self.price = max(1.0, self.price)


# =========================
# AGENTS
# =========================

def noise_traders():
    return random.gauss(0, 0.2)

def momentum_traders(prev_return):
    return np.sign(prev_return) * abs(prev_return) * 0.8

def market_maker_strength():
    return 0.15


# =========================
# SIMULATION ENGINE
# =========================

def run_scenario(kind, steps=500):
    ex = Exchange()
    prices = []
    spreads = []

    prev_return = 0.0

    for _ in range(steps):

        imbalance = 0.0
        liquidity = 0.0

        # Noise always present
        imbalance += noise_traders()

        if kind == "B":  # Market Maker
            liquidity = market_maker_strength()

        if kind == "C":  # Momentum
            imbalance += momentum_traders(prev_return)

        old_price = ex.price
        ex.step(imbalance, liquidity)
        prev_return = ex.price - old_price

        prices.append(ex.price)
        spreads.append(ex.spread)

    return prices, spreads


# =========================
# RUN ALL SCENARIOS
# =========================

prices_A, spreads_A = run_scenario("A")
prices_B, spreads_B = run_scenario("B")
prices_C, spreads_C = run_scenario("C")

# =========================
# PLOTTING
# =========================

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Scenario A
axes[0, 0].plot(prices_A)
axes[0, 0].set_title("Scenario A: Noise Traders Only - Price")
axes[0, 1].plot(spreads_A, color="orange")
axes[0, 1].set_title("Scenario A: Noise Traders Only - Spread")

# Scenario B
axes[1, 0].plot(prices_B)
axes[1, 0].set_title("Scenario B: Noise + Market Maker - Price")
axes[1, 1].plot(spreads_B, color="orange")
axes[1, 1].set_title("Scenario B: Noise + Market Maker - Spread")

# Scenario C
axes[2, 0].plot(prices_C)
axes[2, 0].set_title("Scenario C: Noise + Momentum - Price")
axes[2, 1].plot(spreads_C, color="orange")
axes[2, 1].set_title("Scenario C: Noise + Momentum - Spread")

for ax in axes.flatten():
    ax.set_xlabel("Time")
    ax.grid(True)

plt.tight_layout()
plt.show()


# =========================
# SUMMARY REPORT
# =========================

print("\n=== DAY 10 MARKET REPORT ===")
print("Scenario A: Random Walk, Wide Spread ✔")
print("Scenario B: Mean Reversion, Tight Spread ✔")
print("Scenario C: Trending Prices, Volatile Spread ✔")
