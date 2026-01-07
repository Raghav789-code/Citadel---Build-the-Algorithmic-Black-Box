import heapq
import itertools
from day2_matching_engine import MatchingEngine
from market_sim.day1_order_book import Order


class Event:
    def __init__(self, timestamp, action):
        self.timestamp = timestamp
        self.action = action  # callable

    def execute(self):
        self.action()


class MarketEngine:
    def __init__(self):
        self.time = 0.0
        self.event_queue = []
        self._seq = itertools.count()
        self.exchange = MatchingEngine()

    def schedule(self, timestamp, action):
        heapq.heappush(
            self.event_queue,
            (timestamp, next(self._seq), Event(timestamp, action))
        )

    def run(self):
        while self.event_queue:
            timestamp, _, event = heapq.heappop(self.event_queue)
            self.time = timestamp
            event.execute()
