#!/bin/bash
# Runs the final, full-rep benchmark for the `best` candidate against the
# triton reference and the production (existing _C.so) kernel. Writes a
# clean three-column table to stdout.
#
# Used to populate best.txt.

set -e
HERE=$(dirname "$(readlink -f "$0")")
cd "$HERE"

BATCH_VALS=(1 4 8)
N_CTX_VALS=(128 256 384 512 640 768 1024)

printf "%4s %6s  %18s  %18s  %18s  %8s  %8s\n" \
    "B" "N_CTX" "triton (TFLOP/s)" "prod-tk (TFLOP/s)" "best (TFLOP/s)" "best/tri" "best/prod"

for B in "${BATCH_VALS[@]}"; do
    for N in "${N_CTX_VALS[@]}"; do
        OUT=$(timeout 120 python3 -u bench_one.py $B $N best 2>&1 | tail -1)
        tri=$(echo "$OUT" | awk '{print $2}')
        prod=$(echo "$OUT" | awk '{print $4}')
        best=$(echo "$OUT" | awk '{print $6}')
        ratio_tri=$(python3 -c "print(f'{$best/$tri:.2f}x')")
        ratio_prod=$(python3 -c "print(f'{$best/$prod:.2f}x')")
        printf "%4d %6d  %18s  %18s  %18s  %8s  %8s\n" \
            "$B" "$N" "$tri" "$prod" "$best" "$ratio_tri" "$ratio_prod"
    done
done
