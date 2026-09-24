#!/bin/zsh
# Fase 6c (reports/phase6c-protocol.md), en serie: entrenar A4v4 y A2v4, calibrar con calib12,
# test final final13 y diagnóstico de K. Salida en reports/phase6c/.
set -u
G=.venv/bin/gso
R=reports/phase6c
for c in e4b_v4 e2b_heads_v4; do
  echo "== train $c $(date -u +%H:%M:%S)"
  caffeinate -i $G train --config configs/$c.yaml > $R/train_$c.log 2>&1; echo "EXIT train_$c $?"
done
typeset -A CK
CK=(a4v4 $(ls -d runs/e4b_v4/*/ | tail -1) a2v4 $(ls -d runs/e2b_heads_v4/*/ | tail -1) a4v3 runs/e4b_experiment/20260923T204945Z/)
for m in a4v4 a2v4 a4v3; do echo "ckpt $m ${CK[$m]}"; done
for m in a4v4 a2v4 a4v3; do
  echo "== calibrate $m $(date -u +%H:%M:%S)"
  caffeinate -i $G calibrate --checkpoint ${CK[$m]}checkpoint --split all --dataset data/pilot_v4_calib12 > $R/calibrate_$m.log 2>&1
  echo "EXIT calibrate_$m $?"
done
for m in a4v4 a2v4 a4v3; do
  cal=$(.venv/bin/python -c "import json;t=open('$R/calibrate_$m.log').read();print(json.loads(t[t.index('{'):])['path'])")
  for d in final13 kdiag11_K kdiag11_K8; do
    echo "== eval $m $d $(date -u +%H:%M:%S)"
    extra=(); [ $d = final13 ] && extra=(--baselines)
    caffeinate -i $G evaluate --checkpoint ${CK[$m]}checkpoint --split all --dataset data/pilot_v4_$d --calibration $cal $extra > $R/eval_${d}_$m.log 2>&1
    echo "EXIT eval_${d}_$m $?"
  done
done
echo "ALL DONE $(date -u +%H:%M:%S)"
