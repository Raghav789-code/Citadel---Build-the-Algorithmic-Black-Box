import random
import heapq
import numpy as np
from dataclasses import dataclass
from market_sim.day6_agents import Agent, Action

# =====================================================
# GLOBAL SETTINGS
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LAMBDA_ARRIVAL = 2.0   # average 2 orders per unit time
SIGMA_FAIR = 0.5       # volatility of fair value
SIM_TIME = 50.0        # total simulation time

# =====================================================
# FAIR VALUE (BROWNIAN MOTION)
# =====================================================
class FairValueProcess:
    def __init__(self, start_price=100.0, sigma=0.5):
        self.value = start_price
        self.sigma = sigma

    def step(self):
        self.value += np.random.normal(0, self.sigma)
        self.value = max(self.value, 1.0)
        return self.value

# =====================================================
# NOISE TRADER (ZERO-INTELLIGENCE)
# =====================================================
class NoiseTrader(Agent):
    def get_action(self, market_snapshot: dict) -> Action | None:
        side = "BUY" if random.random() < 0.5 else "SELL"
        quantity = random.randint(1, 10)

        # Noise traders ignore book, use fair value
        fair_value = market_snapshot["fair_value"]

        # Aggressive but random pricing
        price_offset = random.uniform(0.1, 1.0)

        if side == "BUY":
            price = fair_value + price_offset
        else:
            price = fair_value - price_offset

        return Action(
            side=side,
            quantity=quantity,
            order_type="LIMIT",
            price=round(price, 2)
        )

# =====================================================
# EVENT-DRIVEN SIMULATION (POISSON ARRIVALS)
# =====================================================
@dataclass(order=True)
class Event:
    time: float
    agent: Agent

def run_day7_simulation():
    current_time = 0.0
    fair_value = FairValueProcess(sigma=SIGMA_FAIR)

    # Create noise traders
    agents = [NoiseTrader(f"NT_{i}") for i in range(10)]

    # Event queue (min-heap)
    event_queue = []

    # Schedule first event for each agent
    for agent in agents:
        dt = np.random.exponential(1 / LAMBDA_ARRIVAL)
        heapq.heappush(event_queue, Event(current_time + dt, agent))

    print("=== DAY 7 SIMULATION START ===")

    while event_queue and current_time < SIM_TIME:
        event = heapq.heappop(event_queue)
        current_time = event.time

        # Advance fair value
        fv = fair_value.step()

        snapshot = {
            "fair_value": fv
        }

        action = event.agent.get_action(snapshot)

        if action:
            print(
                f"t={current_time:5.2f} | "
                f"{event.agent.id} | "
                f"{action.side} {action.quantity} @ {action.price}"
            )

        # Schedule agent's next arrival
        dt = np.random.exponential(1 / LAMBDA_ARRIVAL)
        heapq.heappush(event_queue, Event(current_time + dt, event.agent))

    print("=== DAY 7 SIMULATION END ===")

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    run_day7_simulation()
