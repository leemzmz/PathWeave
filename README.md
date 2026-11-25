# PathWeave

**English** | [简体中文](README.zh-CN.md)

**PathWeave: A Path-Aware Structural Representation for Macromodel Topology Recovery of Analog Circuits** has been accepted at **DATE 2027**.

PathWeave encodes transistor-level circuits into 128-dimensional embeddings and aligns them with Gm+RC functional macromodels. An optional GRU decoder evaluates topology reconstruction.

- Input representations: Netlist, Vpath-only, and PathWeave.
- GNN backbones: GraphSAGE, GAT, and GINE.
- Evaluation: 42 macromodels with 20 circuits each, 2/4/6/8/10-shot settings, and six random seeds.
- Supports independent training, checkpoint reload evaluation, per-configuration exports, and aggregation across seeds.

## Installation

- Verified environment: Linux/WSL, Python 3.11.13.
- Encoder evaluation with the provided preprocessed graphs runs on CPU without CktPath. Strict decoder comparisons use the original CUDA device by default.

```bash
git clone https://github.com/leemzmz/PathWeave.git app_pathweave
cd app_pathweave
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

- `requirements-dev.txt`: runtime dependencies and test tools.
- `requirements-analysis.txt`: additional dependencies for paper figures and shortcut analyses.
- For CPU decoder diagnostics, use `--decoder-device cpu --allow-differences`. Numerical differences may change generated sequences; exact agreement with the CUDA archive is not guaranteed.

## CktPath: optional private dependency and related papers

- [CktPath](https://github.com/leemzmz/CktPath) is a private repository, available upon request from the authors.
- Evaluating the provided models with the preprocessed graphs does not require CktPath. The public data does not contain its source code.
- CktPath is required for processing new netlists, generating circuits, or running the original training workflow.
- Once access is granted, place it in a sibling `path/` directory and use commit `14938f288c5dcd94f939b02970c7cf15a5505890`.

The CktPath method is also applied in the following two journal papers in **IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (IEEE TCAD)**:

1. Mingzhen Li and Guoyong Shi, [**A Signal-Path Recognition Approach to Multistage Op-Amp Pole-Zero Extraction With Applications**](https://ieeexplore.ieee.org/document/11270852), **IEEE TCAD**, 2026. DOI: [10.1109/TCAD.2025.3638265](https://doi.org/10.1109/TCAD.2025.3638265).
2. Yapeng Li, Peng Xu, Mingzhen Li, Tsung-Yi Ho, Bei Yu, and Tinghuan Chen, [**Analog Circuit Representation Learning Informed by Kirchhoff’s Current Law**](https://ieeexplore.ieee.org/document/11614522), **IEEE TCAD**, 2026. DOI: [10.1109/TCAD.2026.3715070](https://doi.org/10.1109/TCAD.2026.3715070).

## Data and results

| Input | Preprocessed graphs | Models and results |
|---|---|---|
| `_data/` | `preprocessed/data/` | `_run/models/`, `_run/results/` |
| `_data_new/` | `preprocessed/data_new/` | `_run_new/models/`, `_run_new/results/` |

- Each dataset contains 42 macromodels × 20 circuits. The new dataset includes 255 folded and 585 non-folded circuits.
- Both archives contain only `README.md`, `models/`, and `results/`; they do not include training-source or CktPath snapshots.
- Each complete experiment includes 270 circuit encoders, 54 decoders, and six target encoders.
- Tables, figures, and numerical results for the new dataset are in the [experiment report](_run_new/results/analysis/paper_results.md) (Chinese).

## Evaluate the models without CktPath

Start with one configuration on the new dataset:

```bash
python Run/evaluate_preprocessed.py \
  --seeds 42 --shots 2 --backbones graphsage --backends graph_path_attention \
  --output _experiments/reload_example.json
```

Evaluate the complete six-seed matrix on the new dataset (270 encoders and 54 decoders; the decoder requires CUDA):

```bash
python Run/evaluate_preprocessed.py --decoder --output _experiments/reload_new.json
```

Evaluate the original dataset and record differences from the archived results:

```bash
python Run/evaluate_preprocessed.py \
  --dataset _data --run _run --graphs preprocessed/data \
  --decoder --allow-differences --output _experiments/reload_original.json
```

- `preprocessed/data/` and `preprocessed/data_new/` each provide the three input representations for 840 circuits. They contain tensors, feature definitions, and hashes, with no CktPath source code.
- The evaluator loads the weights, recomputes metrics, and compares them with the archived results. It does not train models or modify historical protocols or results.
- Comparisons are strict by default and fail on differences. `--allow-differences` produces a diagnostic report; it does not mark mismatches as successful reproduction.
- Original netlists are used only to verify sample identities and hashes. This entry point does not extract paths from netlists.
- The output must be a new file outside the input and archive directories.
- Full validation in the recorded environment: all metrics for the new dataset's 270 encoders and 54 decoders exactly match the archive.
- For the original dataset, the main retrieval metrics and all 54 decoder results match. Fourteen encoder configurations differ in `target_manifold_spearman`, by at most approximately `2.67e-5`; the command above therefore uses diagnostic mode.
- See the [reproducibility guide](_docs/reproducibility.md) (Chinese) for formats, validation results, and reproduction limits.

## Archive checks and training

Validate existing archives without modifying them (no CktPath required):

```bash
python Run/check_archive.py
PATHWEAVE_DATASET_DIR=_data_new PATHWEAVE_RUN_DIR=_run_new python Run/check_archive.py
```

Start an independent experiment with the current code:

```bash
PATHWEAVE_DATASET_DIR=_data_new \
PATHWEAVE_RUN_DIR=_experiments/example \
./run_dual_pathweave_experiment.sh \
  --phase all --stage encoder --baseline pathweave \
  --shot 2 --gnn graphsage --seed 42
```

- The training workflow reads `_data/` and writes to `_run/` by default; the two environment variables select other dataset and output roots.
- Relative configuration paths are resolved against the project root. Independent experiments write to their own `models/` and `results/` directories.
- `--phase all` runs training, independent reload evaluation, and aggregation in sequence. `--stage decoder` runs the separate 10-shot decoder workflow.
- Each archive corresponds to its original training version. Do not refresh protocols to resume historical checkpoints with the current code.
- The preprocessed-graph entry point recomputes metrics from existing weights; the original netlist training entry point still requires CktPath.
- Successful reload evaluation does not imply that training from scratch reproduces historical weights bit for bit. Do not modify existing protocols to bypass provenance differences.

## Documentation and contributions

The detailed guides below are currently in Chinese:

- [Documentation index](_docs/README.md): architecture, models, datasets, terminology, and reproducibility.
- [Directory structure](_docs/directory_structure.md): module responsibilities and file locations.
- [Contribution guide](CONTRIBUTING.md): checks, archive conventions, and Git release checks.

## License

- Project code is licensed under the [MIT License](LICENSE).
- Third-party dependencies retain their respective licenses and copyright notices.
