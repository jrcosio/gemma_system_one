#!/bin/zsh
# Fase 6d (reports/phase6d-protocol.md): 3 modelos × 5 conjuntos, temperaturas de calib12 (fase 6c).
set -u
G=.venv/bin/gso
R=reports/phase6d
typeset -A CK
CK=(a4v3 runs/e4b_experiment/20260923T204945Z a4v4 runs/e4b_v4/20260924T140932Z a2v4 runs/e2b_heads_v4/20260924T142144Z)
for m in a4v3 a4v4 a2v4; do
  cal=$(.venv/bin/python -c "import json;t=open('reports/phase6c/calibrate_$m.log').read();print(json.loads(t[t.index('{'):])['path'])")
  for d in kdiag14_K4 kdiag14_K8 kdiag14_K4swap kdiag14_K8swap final13_faultswap; do
    echo "== eval $m $d $(date -u +%H:%M:%S)"
    caffeinate -i $G evaluate --checkpoint ${CK[$m]}/checkpoint --split all --dataset data/pilot_v4_$d --calibration $cal > $R/eval_${d}_$m.log 2>&1
    echo "EXIT eval_${d}_$m $?"
  done
done
echo "ALL DONE $(date -u +%H:%M:%S)"
