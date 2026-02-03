import random
import heapq
import numpy as np
from collections import deque
from dataclasses import dataclass
from market_sim.day6_agents import Agent, Action

# =====================================================
# GLOBAL SETTINGS
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LAMBDA_ARRIVAL = 2.0
SIGMA_FAIR = 0.3
SIM_TIME = 30.0
WINDOW = 10        # lookback window for momentum

# =====================================================
# FAIR VALUE (FOR REFERENCE ONLY)
# =====================================================
class FairValueProcess:
    def __init__(self, start_price=100.0, sigma=0.3):
        self.value = start_price
        self.sigma = sigma

    def step(self):
        self.value += np.random.normal(0, self.sigma)
        self.value = max(self.value, 1.0)
        return self.value

# =====================================================
# MOMENTUM AGENT
# =====================================================
class MomentumAgent(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.prices = deque(maxlen=WINDOW)

    def get_action(self, market_snapshot: dict) -> Action | None:
        price = market_snapshot["last_price"]
        self.prices.append(price)

        # Need enough history
        if len(self.prices) < WINDOW:
            return None

        sma_old = np.mean(list(self.prices)[:-1])
        sma_new = np.mean(self.prices)

        if sma_new > sma_old:
            side = "BUY"
        elif sma_new < sma_old:
            side = "SELL"
        else:
            return None

        quantity = random.randint(5, 15)

        return Action(
            side=side,
            quantity=quantity,
            order_type="MARKET"
        )

# =====================================================
# EVENT SYSTEM
# =====================================================
@dataclass(order=True)
class Event:
    time: float
    agent: Agent

# =====================================================
# DAY 8 SIMULATION (100% MOMENTUM)
# =====================================================
def run_day8_simulation():
    current_time = 0.0
    fair_value = FairValueProcess()
    last_price = fair_value.value

    agents = [MomentumAgent(f"M_{i}") for i in range(15)]
    event_queue = []

    for agent in agents:
        dt = np.random.exponential(1 / LAMBDA_ARRIVAL)
        heapq.heappush(event_queue, Event(current_time + dt, agent))

    print("=== DAY 8 MOMENTUM SIMULATION START ===")

    while event_queue and current_time < SIM_TIME:
        event = heapq.heappop(event_queue)
        current_time = event.time

        # Underlying price evolves
        last_price = fair_value.step()

        snapshot = {
            "last_price": last_price
        }

        action = event.agent.get_action(snapshot)

        if action:
            direction = "↑" if action.side == "BUY" else "↓"
            print(
                f"t={current_time:5.2f} | "
                f"{event.agent.id} | "
                f"{action.side} {action.quantity} {direction} @ {last_price:.2f}"
            )

            # Feedback loop (THIS CAUSES THE PUMP)
            if action.side == "BUY":
                last_price += 0.5
            else:
                last_price -= 0.5

        dt = np.random.exponential(1 / LAMBDA_ARRIVAL)
        heapq.heappush(event_queue, Event(current_time + dt, event.agent))

    print("=== DAY 8 MOMENTUM SIMULATION END ===")

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    run_day8_simulation()
