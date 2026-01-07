# Citadel — Build the Algorithmic Black Box  
**An Event-Driven Market Microstructure Simulator with Agent Ecology**

---

## Overview

**Citadel** is a research-grade, event-driven market simulator that recreates the internal mechanics of a modern electronic exchange.  
It models a **central limit order book (CLOB)**, a **deterministic matching engine**, a **discrete-event scheduler**, and a heterogeneous **ecosystem of trading agents** whose interactions generate realistic market dynamics.

This project is designed to move from **first principles of market microstructure** to **emergent macro behavior**, with strict correctness, reproducibility, and validation guarantees.

By the end of execution, the simulator can reproduce **three distinct market regimes** and automatically generate a **multi-page PDF market report**.

---

## Key Features

- Event-driven exchange engine (no `sleep`, no wall-clock time)
- FIFO, price-time priority matching engine
- Limit & market orders with partial fills
- Deterministic replay with fixed random seeds
- Trade tape (Time & Sales), L1 & L2 snapshots
- VWAP, spread, and volatility analytics
- Agent-based market ecology:
  - Noise Traders
  - Momentum (Trend-Following) Traders
  - Market Makers with inventory skew
- Automatic validation via assertions
- One-command reproducibility

---

## Project Philosophy

> **Markets are not prices — they are processes.**

This simulator emphasizes:
- **Structure over intelligence**
- **Rules over prediction**
- **Emergence over hard-coding**

No agent has access to the future.  
No agent touches the order book directly.  
All realism emerges from interaction under shared constraints.

---
---

## License

Educational & research use only.
