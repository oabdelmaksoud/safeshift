# SafeShift — Scalability (analysis runtime vs size)

Median of 3 runs; sparse layered architectures (interfaces ~ 2.5x components). The main sweep uses near-acyclic graphs (short back-edges); the cyclic stress row below uses dense back-edges to any earlier layer to force genuine directed cycles.

| Components | Interfaces | Median analysis time (s) |
|-----------:|-----------:|-------------------------:|
| 20 | 33 | 0.0004 |
| 50 | 83 | 0.0011 |
| 100 | 187 | 0.0030 |
| 200 | 366 | 0.0083 |
| 500 | 956 | 0.0366 |

**Cyclic stress (SCC-based cycle membership):** 500 components, 1026 interfaces, 128 nodes on directed cycles -- analysed in **0.1132 s**. Because membership uses strongly-connected components (O(V+E)) rather than cycle enumeration, dense cyclic graphs stay fast.

Figure: `figures/scalability.png`.