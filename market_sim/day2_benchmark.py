# benchmark.py
import time
import heapq
import random

N = 10_000
prices = [random.uniform(90, 110) for _ in range(N)]

# LIST
lst = []
t0 = time.time()
for p in prices:
    lst.append(p)
    lst.sort()
print("List time:", time.time() - t0)

# HEAP
heap = []
t0 = time.time()
for p in prices:
    heapq.heappush(heap, p)
print("Heap time:", time.time() - t0)
