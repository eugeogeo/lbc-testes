#!/usr/bin/env bash
# Roda todos os experimentos e gera as provas visuais para FMD, KTH-TIPS2b e DTD.
# Uso: ./run_all.sh [fmd kth dtd]   (sem argumentos, roda as três bases)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-venv/bin/python}"
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(fmd kth dtd)
DTD_LABELS="dataset_texture/test/DTD/dtd/labels"

mkdir -p results saved_models logs

for ds in "${DATASETS[@]}"; do
    if [ "$ds" = "dtd" ]; then
        missing=0
        for n in $(seq 1 10); do
            for part in train val test; do
                [ -f "$DTD_LABELS/${part}${n}.txt" ] || missing=1
            done
        done
        if [ "$missing" -eq 1 ]; then
            echo "ERRO: splits oficiais da DTD ausentes em $DTD_LABELS (train/val/test 1..10)." >&2
            echo "Copie a pasta 'labels' do pacote dtd-r1.0.1 e rode: ./run_all.sh dtd" >&2
            continue
        fi
    fi

    echo "=== Base: $ds ==="
    {
        "$PYTHON" resnet_baseline.py --dataset "$ds"
        "$PYTHON" train_resnet_scratch_cv.py --dataset "$ds"
        "$PYTHON" train_integrated_cv.py --dataset "$ds"
        "$PYTHON" train_integrated_scratch_cv.py --dataset "$ds"
        "$PYTHON" plot_filters.py --dataset "$ds"
        "$PYTHON" plot_confusion.py --dataset "$ds"
    } 2>&1 | tee "logs/run_${ds}.log"
done

# O baseline handcrafted só suporta a FMD.
if [[ " ${DATASETS[*]} " == *" fmd "* ]]; then
    "$PYTHON" handcrafted_baseline.py 2>&1 | tee logs/run_handcrafted_fmd.log
fi
