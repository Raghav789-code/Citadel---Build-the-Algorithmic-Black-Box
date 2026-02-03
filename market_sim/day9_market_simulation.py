import random
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =====================
# BASIC DATA STRUCTURES
# =====================

@dataclass
class Order:
    side: str      # "BUY" or "SELL"
    price: float
    qty: int


# =====================
# SIMPLE EXCHANGE
# =====================

class Exchange:
    def __init__(self, start_price=100.0):
        self.price = start_price
        self.trades = []

    def submit(self, order: Order):
        # Simple market impact model
        impact = 0.002 * order.qty

        if order.side == "BUY":
            self.price += impact
        else:
            self.price -= impact

        self.trades.append(self.price)

    def mid_price(self):
        return self.price


# =====================
# MOMENTUM AGENT
# =====================

class MomentumAgent:
    def __init__(self):
        self.last_price = None

    def act(self, price):
        if self.last_price is None:
            self.last_price = price
            return None

        side = "BUY" if price > self.last_price else "SELL"
        self.last_price = price

        return Order(
            side=side,
            price=price,
            qty=random.randint(1, 10)
        )


# =====================
# MARKET MAKER
# =====================

class MarketMaker:
    def __init__(self):
        self.inventory = 0
        self.cash = 0.0
        self.fair_price = 100.0

        self.base_spread = 0.2
        self.inv_sensitivity = 0.01
        self.max_inventory = 50

    def quote(self, market_price):
        # Update fair value (anchor)
        self.fair_price = 0.95 * self.fair_price + 0.05 * market_price

        # Inventory-adjusted mid price
        skew = self.inventory * self.inv_sensitivity
        mid = 0.8 * self.fair_price + 0.2 * market_price - skew

        # Dynamic spread widens with inventory risk
        spread = self.base_spread + abs(self.inventory) * 0.005

        bid = mid - spread / 2
        ask = mid + spread / 2

        return bid, ask

    def act(self, market_price):
        bid, ask = self.quote(market_price)

        # Enforce inventory limits
        if self.inventory > self.max_inventory:
            return Order("SELL", ask, random.randint(5, 15))

        if self.inventory < -self.max_inventory:
            return Order("BUY", bid, random.randint(5, 15))

        # Otherwise quote both sides randomly
        side = random.choice(["BUY", "SELL"])
        price = bid if side == "BUY" else ask

        return Order(side, price, random.randint(1, 10))

    def fill(self, order: Order):
        if order.side == "BUY":
            self.inventory += order.qty
            self.cash -= order.qty * order.price
        else:
            self.inventory -= order.qty
            self.cash += order.qty * order.price

    def pnl(self, market_price):
        return self.cash + self.inventory * market_price


# =====================
# SIMULATION
# =====================

def run_simulation(steps=1000, with_mm=True):
    ex = Exchange()
    momentum = MomentumAgent()
    mm = MarketMaker() if with_mm else None

    prices = []

    for _ in range(steps):
        # Momentum trader acts
        mo = momentum.act(ex.mid_price())
        if mo:
            ex.submit(mo)

        # Market maker acts
        if with_mm:
            mm_order = mm.act(ex.mid_price())
            ex.submit(mm_order)
            mm.fill(mm_order)

        prices.append(ex.mid_price())

    return prices, mm


# =====================
# RUN + PLOT
# =====================

if __name__ == "__main__":
    prices_mm, mm = run_simulation(with_mm=True)
    prices_mom, _ = run_simulation(with_mm=False)

    plt.figure(figsize=(12, 6))
    plt.plot(prices_mm, label="With Market Maker", linewidth=2)
    plt.plot(prices_mom, linestyle="--", label="Momentum Only")

    plt.title("Momentum vs Market Maker (Stabilized)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Final Market Maker Inventory:", mm.inventory)
    print("Final Market Maker PnL:", round(mm.pnl(prices_mm[-1]), 2))
