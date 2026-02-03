import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

# =====================================================
# ACTION DEFINITIONS (INTENTS, NOT EXECUTION)
# =====================================================

@dataclass(frozen=True)
class Action:
    """
    Immutable trading intent emitted by an agent.
    The exchange will interpret and execute it.
    """
    side: str        # "BUY" or "SELL"
    quantity: int    # positive integer
    order_type: str  # "MARKET" or "LIMIT"
    price: float | None = None


# =====================================================
# AGENT BASE CLASS (ABSTRACT)
# =====================================================

class Agent(ABC):
    """
    Abstract base class for all agents.
    Enforces a common interface.
    """

    def __init__(self, agent_id: str, cash: float = 10_000, inventory: int = 0):
        self.id = agent_id
        self.cash = cash
        self.inventory = inventory

    @abstractmethod
    def get_action(self, market_snapshot: dict) -> Action | None:
        """
        Decide what to do based on current market snapshot.

        Must return:
        - An Action object, OR
        - None (no-op)
        """
        pass


# =====================================================
# RANDOM AGENT (DAY 6 EXERCISE)
# =====================================================

class RandomAgent(Agent):
    """
    Buys or sells randomly with 50/50 probability.
    Ignores market state entirely.
    """

    def get_action(self, market_snapshot: dict) -> Action | None:
        side = "BUY" if random.random() < 0.5 else "SELL"
        quantity = random.randint(1, 10)

        return Action(
            side=side,
            quantity=quantity,
            order_type="MARKET"
        )
if __name__ == "__main__":
    agent = RandomAgent("R1")
    snapshot = {"best_bid": 99.5, "best_ask": 100.5}

    for _ in range(5):
        print(agent.get_action(snapshot))
