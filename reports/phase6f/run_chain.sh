#!/bin/zsh
# Fase 6f (reports/phase6f-protocol.md), en serie: entrenar A4v5 y A2v5, calibrar con calib16,
# test final17 y tríos trip18_K4. Salida en reports/phase6f/.
set -u
G=.venv/bin/gso
R=reports/phase6f
for c in e4b_v5 e2b_heads_v5; do
  echo "== train $c $(date -u +%H:%M:%S)"
  caffeinate -i $G train --config configs/$c.yaml > $R/train_$c.log 2>&1; echo "EXIT train_$c $?"
done
typeset -A CK
CK=(a4v5 $(ls -d runs/e4b_v5/*/ | tail -1) a2v5 $(ls -d runs/e2b_heads_v5/*/ | tail -1) a4v3 runs/e4b_experiment/20260923T204945Z/)
for m in a4v5 a2v5 a4v3; do echo "ckpt $m ${CK[$m]}"; done
for m in a4v5 a2v5 a4v3; do
  echo "== calibrate $m $(date -u +%H:%M:%S)"
  caffeinate -i $G calibrate --checkpoint ${CK[$m]}checkpoint --split all --dataset data/pilot_v5_calib16 > $R/calibrate_$m.log 2>&1
  echo "EXIT calibrate_$m $?"
done
for m in a4v5 a2v5 a4v3; do
  cal=$(.venv/bin/python -c "import json;t=open('$R/calibrate_$m.log').read();print(json.loads(t[t.index('{'):])['path'])")
  for d in final17 trip18_K4; do
    echo "== eval $m $d $(date -u +%H:%M:%S)"
    extra=(); [ $d = final17 ] && extra=(--baselines)
    caffeinate -i $G evaluate --checkpoint ${CK[$m]}checkpoint --split all --dataset data/pilot_v5_$d --calibration $cal $extra > $R/eval_${d}_$m.log 2>&1
    echo "EXIT eval_${d}_$m $?"
  done
done
echo "ALL DONE $(date -u +%H:%M:%S)"
