#!/bin/zsh
# Fase 6e (reports/phase6e-protocol.md): 3 modelos × {K4, K8}, temperaturas de calib12 (fase 6c).
set -u
G=.venv/bin/gso
R=reports/phase6e
typeset -A CK
CK=(a4v3 runs/e4b_experiment/20260923T204945Z a4v4 runs/e4b_v4/20260924T140932Z a2v4 runs/e2b_heads_v4/20260924T142144Z)
for m in a4v3 a4v4 a2v4; do
  cal=$(.venv/bin/python -c "import json;t=open('reports/phase6c/calibrate_$m.log').read();print(json.loads(t[t.index('{'):])['path'])")
  for d in trip15_K4 trip15_K8; do
    echo "== eval $m $d $(date -u +%H:%M:%S)"
    caffeinate -i $G evaluate --checkpoint ${CK[$m]}/checkpoint --split all --dataset data/pilot_v4_$d --calibration $cal > $R/eval_${d}_$m.log 2>&1
    echo "EXIT eval_${d}_$m $?"
  done
done
echo "ALL DONE $(date -u +%H:%M:%S)"
