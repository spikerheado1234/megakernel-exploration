#!/bin/bash
# Runs bench_one over the full sweep, each in a fresh process to avoid
# the cross-shape hang we see when triton+prod+candidate all run in one
# process. Saves a clean table to stdout.
#
# Usage:
#   ./bench_sweep.sh v2_kv64 [v3_xxx ...]
#
# Output columns: B N_CTX  triton  prod  cand1  cand2 ...

set -e
HERE=$(dirname "$(readlink -f "$0")")
cd "$HERE"

CANDS=("$@")
if [ ${#CANDS[@]} -eq 0 ]; then
    echo "usage: $0 <cand1> [cand2 ...]"
    exit 2
fi

BATCH_VALS=(1 4 8)
N_CTX_VALS=(128 256 384 512 640 768 1024)

# Header
printf "%4s %6s  %8s  %8s" "B" "N_CTX" "triton" "prod-tk"
for c in "${CANDS[@]}"; do
    printf "  %12s" "$c"
done
printf "\n"

for B in "${BATCH_VALS[@]}"; do
    for N in "${N_CTX_VALS[@]}"; do
        OUT=$(timeout 90 python3 -u bench_one.py $B $N "${CANDS[0]}" 2>&1 | tail -1)
        tri=$(echo "$OUT" | awk '{print $2}')
        prod=$(echo "$OUT" | awk '{print $4}')
        cand1=$(echo "$OUT" | awk '{print $6}')
        printf "%4d %6d  %8s  %8s  %12s" $B $N "$tri" "$prod" "$cand1"
        for ((i=1; i<${#CANDS[@]}; i++)); do
            c="${CANDS[$i]}"
            OUT2=$(timeout 90 python3 -u bench_one.py $B $N "$c" 2>&1 | tail -1)
            cv=$(echo "$OUT2" | awk '{print $6}')
            printf "  %12s" "$cv"
        done
        printf "\n"
    done
done
