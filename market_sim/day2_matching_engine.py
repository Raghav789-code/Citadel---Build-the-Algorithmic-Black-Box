import heapq
from market_sim.day1_order_book import Order, OrderBook


class MatchingEngine(OrderBook):
    """
    Extends OrderBook with matching logic.
    Implements price-time priority, market orders, and partial fills.
    """

    def __init__(self):
        super().__init__()
        self.trades = []  # trade tape

    # -------------------------------------------------
    # Public entry point
    # -------------------------------------------------
    def add_order(self, order: Order):
        """
        Add an incoming order:
        1. Try to match immediately
        2. If LIMIT and remaining > 0, add to book
        """
        self.match(order)

        if order.remaining > 0 and order.type == "LIMIT":
            self._add_to_book(order)

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------
    def _add_to_book(self, order: Order):
        seq = next(self._seq)

        if order.side == "BUY":
            heapq.heappush(self.bids, (-order.price, seq, order))
        else:
            heapq.heappush(self.asks, (order.price, seq, order))

    def match(self, incoming: Order):
        if incoming.side == "BUY":
            self._match_buy(incoming)
        else:
            self._match_sell(incoming)

    # -------------------------------------------------
    # BUY side matching
    # -------------------------------------------------
    def _match_buy(self, buy: Order):
        while buy.remaining > 0 and self.asks:
            best_price, _, ask = self.asks[0]

            # LIMIT order price check
            if buy.type == "LIMIT" and buy.price < best_price:
                break

            trade_qty = min(buy.remaining, ask.remaining)
            trade_price = ask.price  # resting price rule

            self._record_trade(
                price=trade_price,
                qty=trade_qty,
                buyer=buy.id,
                seller=ask.id
            )

            buy.remaining -= trade_qty
            ask.remaining -= trade_qty

            if ask.remaining == 0:
                ask.status = "FILLED"
                heapq.heappop(self.asks)
            else:
                ask.status = "PARTIAL"

        self._update_status(buy)

    # -------------------------------------------------
    # SELL side matching
    # -------------------------------------------------
    def _match_sell(self, sell: Order):
        while sell.remaining > 0 and self.bids:
            best_price, _, bid = self.bids[0]
            best_price = -best_price

            # LIMIT order price check
            if sell.type == "LIMIT" and sell.price > best_price:
                break

            trade_qty = min(sell.remaining, bid.remaining)
            trade_price = bid.price  # resting price rule

            self._record_trade(
                price=trade_price,
                qty=trade_qty,
                buyer=bid.id,
                seller=sell.id
            )

            sell.remaining -= trade_qty
            bid.remaining -= trade_qty

            if bid.remaining == 0:
                bid.status = "FILLED"
                heapq.heappop(self.bids)
            else:
                bid.status = "PARTIAL"

        self._update_status(sell)

    # -------------------------------------------------
    # Trade logging
    # -------------------------------------------------
    def _record_trade(self, price, qty, buyer, seller):
        self.trades.append({
            "price": price,
            "qty": qty,
            "buyer": buyer,
            "seller": seller
        })

    def _update_status(self, order: Order):
        if order.remaining == 0:
            order.status = "FILLED"
        elif order.remaining < order.quantity:
            order.status = "PARTIAL"
        else:
            order.status = "OPEN"


# =====================================================
# NON-NEGOTIABLE VALIDATION TEST (RUNS DIRECTLY)
# =====================================================
if __name__ == "__main__":
    engine = MatchingEngine()

    # Submit ASK ladder
    engine.add_order(Order("A1", "SELL", 10, 1, price=101))
    engine.add_order(Order("A2", "SELL", 20, 2, price=102))
    engine.add_order(Order("A3", "SELL", 30, 3, price=103))

    # Submit MARKET BUY that clears the book
    engine.add_order(Order("B1", "BUY", 60, 4, order_type="MARKET"))

    # Assertions (HARD REQUIREMENTS)
    assert len(engine.trades) == 3
    assert [t["price"] for t in engine.trades] == [101, 102, 103]
    assert not engine.asks
    assert engine.bids == []

    print("ladder test passed")
