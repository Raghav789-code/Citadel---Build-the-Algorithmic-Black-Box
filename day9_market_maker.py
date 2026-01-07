from dataclasses import dataclass
import random
import time

# =========================
# DATA STRUCTURES
# =========================

@dataclass
class Quote:
    bid: float
    ask: float


@dataclass
class Trade:
    time: float
    agent: str
    side: str
    qty: int
    price: float


# =========================
# MARKET MAKER AGENT
# =========================

class MarketMakerAgent:
    def __init__(
        self,
        agent_id: str,
        spread: float = 0.2,
        inventory_limit: int = 100,
        inventory_sensitivity: float = 0.01
    ):
        self.agent_id = agent_id
        self.spread = spread
        self.inventory = 0
        self.inventory_limit = inventory_limit
        self.inventory_sensitivity = inventory_sensitivity

    def quote(self, mid_price: float) -> Quote:
        price_shift = self.inventory * self.inventory_sensitivity

        bid = mid_price - self.spread / 2 - price_shift
        ask = mid_price + self.spread / 2 - price_shift

        return Quote(
            bid=round(bid, 2),
            ask=round(ask, 2)
        )

    def on_trade(self, side: str, quantity: int):
        if side == "BUY":
            self.inventory -= quantity
        elif side == "SELL":
            self.inventory += quantity

        self.inventory = max(
            -self.inventory_limit,
            min(self.inventory, self.inventory_limit)
        )

    def __repr__(self):
        return f"MM({self.agent_id}) | Inventory={self.inventory}"


# =========================
# SIMULATION LOOP
# =========================

def run_simulation(steps=100):
    mm = MarketMakerAgent("MM_1")
    mid_price = 100.0
    trades = []

    print("\n=== MARKET MAKER SIMULATION START ===")

    for _ in range(steps):
        time_stamp = round(time.time() % 1000, 2)

        quote = mm.quote(mid_price)

        side = random.choice(["BUY", "SELL"])
        qty = random.randint(1, 10)

        if side == "BUY":
            trade_price = quote.ask
            mm.on_trade("BUY", qty)
        else:
            trade_price = quote.bid
            mm.on_trade("SELL", qty)

        trades.append(
            Trade(
                time=time_stamp,
                agent=mm.agent_id,
                side=side,
                qty=qty,
                price=trade_price
            )
        )

        mid_price += random.uniform(-0.05, 0.05)

        print(
            f"t={time_stamp:6} | {side:4} {qty:2} @ {trade_price:6} | "
            f"Inventory={mm.inventory:4}"
        )

    print("\n=== SIMULATION END ===")
    return trades


# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_simulation()
