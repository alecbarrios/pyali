#!/bin/bash
# Post-run analysis of a cycle's reruns on the two reference fields: A/B vs the reference results,
# per-stage table, paired SNR, QC figure.
#   e1_post.sh <experiment_root> <reference_plate_root> [variant]
#   e.g. e1_post.sh /scratch/E2 /home/jovyan/spatial-technology-platform/AB/pyali/443screen2/Plate_4 w32
set -u
S=$(cd "$(dirname "$0")" && pwd)
E=${1:?experiment root holding <fov_name>/<variant>/}
D=${2:?reference root holding Well_X/<fov_name>/}
V=${3:-lazy}
PY=${PYTHON:-/home/jovyan/.venvs/pyali/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
cd "$S/../../.."
for spec in "164512_P-4_W-A2_443-1_443screen2_DIV35__burst41:Well_A2" "140913_P-4_W-C3_443-2_443screen2_DIV35__burst45:Well_C3"; do
  F=${spec%%:*}; WELL=${spec##*:}
  echo "=================== $F"
  echo "--- A/B: $V vs reference"
  $PY $S/e1_ab.py $E/$F/$V $D/$WELL/$F --out $E/$F/ab_$V.json --label-a $V --label-b ref 2>&1 | grep -v "^  "
  echo "--- table"
  $PY $S/e1_table.py $E/$F baseline solver lazy $V
  echo "--- paired SNR $V vs reference"
  $PY snr_analysis/snr/snr_compare.py $E/$F/$V $D/$WELL/$F --out $E/$F/snr_${V}_vs_ref --label-a $V --label-b ref > $E/$F/snr_${V}_vs_ref.log 2>&1
  $PY - <<EOF
import json
r = json.load(open("$E/$F/snr_${V}_vs_ref/summary.json")); p = r.get("paired", r)
for m in ("snr_median", "snr_p90", "spectral_hf_snr", "noise_sigma"):
    if m in p: print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p[m].items()})
EOF
  echo "--- QC figure"
  $PY scripts/qc_footprints.py $E/$F/$V 2>&1 | head -1
done
