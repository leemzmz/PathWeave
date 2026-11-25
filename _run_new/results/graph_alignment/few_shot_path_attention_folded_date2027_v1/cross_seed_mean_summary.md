# Cross-Seed Mean Summary

- Seeds: 42, 205, 316, 327, 330, 501.
- Values are mean +/- sample standard deviation across seeds.
- Each seed contributes equally.

| Representation | Encoder Top-1 | Encoder MRR | Target-only exact | Task exact | Task valid | Stage exact | GED |
|---|---:|---:|---:|---:|---:|---:|---:|
| Netlist | 59.48 +/- 1.56% | 72.99 +/- 1.26% | 98.90 +/- 1.29% | 76.60 +/- 2.01% | 96.69 +/- 1.11% | 95.01 +/- 1.06% | 0.9418 +/- 0.0116 |
| Vpath Attention | 68.30 +/- 1.15% | 79.81 +/- 0.80% | 98.90 +/- 1.29% | 77.98 +/- 4.32% | 97.02 +/- 0.38% | 96.14 +/- 0.47% | 0.9467 +/- 0.0072 |
| Path Attention | 75.26 +/- 1.84% | 84.59 +/- 1.35% | 98.90 +/- 1.29% | 82.05 +/- 3.16% | 97.84 +/- 0.65% | 97.02 +/- 0.59% | 0.9603 +/- 0.0042 |

## Per-Seed Task Exact

| Seed | Netlist | Vpath Attention | Path Attention |
|---:|---:|---:|---:|
| 42 | 75.79% | 78.02% | 85.16% |
| 205 | 76.43% | 82.30% | 83.10% |
| 316 | 77.46% | 77.06% | 79.60% |
| 327 | 79.60% | 81.67% | 84.37% |
| 330 | 76.83% | 70.24% | 76.90% |
| 501 | 73.49% | 78.57% | 83.17% |
