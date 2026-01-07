import heapq
import itertools

# ==========================================
# ORDER OBJECT (Backtesting-Grade)
# ==========================================

class Order:
    def __init__(self, order_id, side, quantity, timestamp, price=None, order_type="LIMIT"):
        self.id = order_id
        self.side = side            # "BUY" or "SELL"
        self.price = price          # None for MARKET
        self.quantity = quantity
        self.remaining = quantity
        self.timestamp = timestamp
        self.type = order_type      # "LIMIT" or "MARKET"
        self.status = "OPEN"        # OPEN / PARTIAL / FILLED / CANCELLED

    def __repr__(self):
        return f"[{self.side} {self.id} {self.remaining}/{self.quantity} @ {self.price}]"


# ==========================================
# ORDER BOOK (DATA STRUCTURE ONLY)
# ==========================================

class OrderBook:
    def __init__(self):
        self.bids = []  # (-price, seq, order)
        self.asks = []  # ( price, seq, order)
        self._seq = itertools.count()

    # --------------------
    # ADD ORDER (NO MATCHING)
    # --------------------
    def add_order(self, order: Order):
        seq = next(self._seq)

        if order.side == "BUY":
            if order.price is None:
                raise ValueError("LIMIT price required for Day-1")
            heapq.heappush(self.bids, (-order.price, seq, order))

        elif order.side == "SELL":
            if order.price is None:
                raise ValueError("LIMIT price required for Day-1")
            heapq.heappush(self.asks, (order.price, seq, order))

        else:
            raise ValueError("Order side must be BUY or SELL")

    # --------------------
    # BOOK INSPECTION
    # --------------------
    def best_bid(self):
        return -self.bids[0][0] if self.bids else None

    def best_ask(self):
        return self.asks[0][0] if self.asks else None

    def snapshot(self, depth=5):
        bids = sorted(self.bids, reverse=True)[:depth]
        asks = sorted(self.asks)[:depth]

        return {
            "bids": [( -p, o.remaining ) for p, _, o in bids],
            "asks": [(  p, o.remaining ) for p, _, o in asks],
        }
