# Cross-Seed Mean Summary

- Seeds: 42, 205, 316, 327, 330, 501.
- Values are mean +/- sample standard deviation across seeds.
- Each seed contributes equally.

| Representation | Encoder Top-1 | Encoder MRR | Target-only exact | Task exact | Task valid | Stage exact | GED |
|---|---:|---:|---:|---:|---:|---:|---:|
| Netlist | 62.69 +/- 1.09% | 75.39 +/- 0.98% | 98.68 +/- 1.47% | 79.68 +/- 5.81% | 97.24 +/- 0.87% | 95.29 +/- 1.34% | 0.9524 +/- 0.0104 |
| Vpath Attention | 70.39 +/- 0.72% | 81.17 +/- 0.51% | 98.68 +/- 1.47% | 80.79 +/- 1.75% | 96.43 +/- 1.25% | 95.63 +/- 1.47% | 0.9460 +/- 0.0110 |
| Path Attention | 78.11 +/- 1.77% | 86.64 +/- 1.29% | 98.68 +/- 1.47% | 87.21 +/- 2.94% | 98.28 +/- 0.26% | 98.08 +/- 0.25% | 0.9707 +/- 0.0053 |

## Per-Seed Task Exact

| Seed | Netlist | Vpath Attention | Path Attention |
|---:|---:|---:|---:|
| 42 | 84.29% | 81.27% | 88.97% |
| 205 | 78.81% | 81.51% | 89.13% |
| 316 | 81.75% | 80.40% | 81.35% |
| 327 | 81.98% | 77.46% | 88.25% |
| 330 | 68.41% | 82.30% | 88.17% |
| 501 | 82.86% | 81.83% | 87.38% |
