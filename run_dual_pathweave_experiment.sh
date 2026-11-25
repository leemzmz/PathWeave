#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

baseline="pathweave"
shot="10"
gnn="graphsage"
phase="train"
stage="encoder"
refresh=""
seeds=(42)

usage() {
    cat <<EOF
Usage:
  ./run_dual_pathweave_experiment.sh [options]

Options:
  --baseline netlist|current|pathweave|all          正式三表示，default: pathweave
  --shot 2|4|6|8|10|all                      default: 10
  --gnn graphsage|gat|gin|all                default: graphsage
  --seed N [N ...]                           default: 42
  --stage encoder|decoder|all                default: encoder
  --phase all|train|evaluate|summarize       default: train
  --refresh                                  refresh metadata only before artifacts exist
  -h, --help

Environment:
  PATHWEAVE_DATASET_DIR                     input dataset (default: _data)
  PATHWEAVE_RUN_DIR                         independent output root (default: _run)

Examples:
  ./run_dual_pathweave_experiment.sh --baseline pathweave --shot 10 --gnn graphsage --seed 42
  ./run_dual_pathweave_experiment.sh --stage decoder --seed 42
  ./run_dual_pathweave_experiment.sh --baseline all --shot all --gnn all --seed 42 205 316 327 330 501
EOF
}

while (( $# )); do
    case "$1" in
        --baseline)
            baseline="${2:?missing value for --baseline}"
            shift 2
            ;;
        --shot)
            shot="${2:?missing value for --shot}"
            shift 2
            ;;
        --gnn)
            gnn="${2:?missing value for --gnn}"
            shift 2
            ;;
        --phase)
            phase="${2:?missing value for --phase}"
            shift 2
            ;;
        --stage)
            stage="${2:?missing value for --stage}"
            shift 2
            ;;
        --seed)
            shift
            seeds=()
            while (( $# )) && [[ "$1" != --* ]]; do
                seeds+=("$1")
                shift
            done
            if (( ${#seeds[@]} == 0 )); then
                printf 'Missing value for --seed\n' >&2
                exit 2
            fi
            ;;
        --refresh)
            refresh="--refresh-protocol"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$baseline" in
    netlist) backends=(graph_netlist) ;;
    current) backends=(graph_vpath_attention) ;;
    pathweave) backends=(graph_path_attention) ;;
    all) backends=(graph_netlist graph_vpath_attention graph_path_attention) ;;
    *)
        printf 'Invalid --baseline: %s\n' "$baseline" >&2
        exit 2
        ;;
esac

case "$shot" in
    all) shots=(2 4 6 8 10) ;;
    2|4|6|8|10) shots=("$shot") ;;
    *)
        printf 'Invalid --shot: %s\n' "$shot" >&2
        exit 2
        ;;
esac

case "$gnn" in
    all) backbones=(graphsage gat gin) ;;
    graphsage|gat|gin) backbones=("$gnn") ;;
    *)
        printf 'Invalid --gnn: %s\n' "$gnn" >&2
        exit 2
        ;;
esac

case "$phase" in
    all|train|evaluate|summarize) ;;
    *)
        printf 'Invalid --phase: %s\n' "$phase" >&2
        exit 2
        ;;
esac

case "$stage" in
    encoder|decoder|all) ;;
    *)
        printf 'Invalid --stage: %s\n' "$stage" >&2
        exit 2
        ;;
esac

for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        printf 'Invalid seed: %s\n' "$seed" >&2
        exit 2
    fi
done

printf 'Baseline: %s\n' "$baseline"
printf 'Shots: %s\n' "${shots[*]}"
printf 'GNN: %s\n' "${backbones[*]}"
printf 'Seeds: %s\n' "${seeds[*]}"
printf 'Stage: %s\n' "$stage"
printf 'Phase: %s\n' "$phase"

for seed in "${seeds[@]}"; do
    if [[ "$stage" == "encoder" || "$stage" == "all" ]] && (( ${#backends[@]} )); then
        PYTHONHASHSEED="$seed" python -u \
            Run/train_encoder_few_shot.py \
            --phase "$phase" \
            --seed "$seed" \
            $refresh \
            --shots "${shots[@]}" \
            --backbones "${backbones[@]}" \
            --backends "${backends[@]}"
    fi

    if [[ "$stage" == "decoder" || "$stage" == "all" ]]; then
        PYTHONHASHSEED="$seed" python -u \
            Run/train_shared_task_decoder.py \
            --seed "$seed"
    fi
done
