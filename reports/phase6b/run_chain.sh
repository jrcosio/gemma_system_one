#!/bin/zsh
# Fase 6b (reports/phase6b-protocol.md): calibración externa (calib7) y test final (final8), en serie.
set -u
G=.venv/bin/gso
typeset -A CK
CK=(a2 runs/pilot_ce_v3/20260923T005721Z b2 runs/pilot_lora_v3/20260923T012208Z a4 runs/e4b_experiment/20260923T204945Z)
for m in a2 b2 a4; do
  echo "== calibrate $m $(date -u +%H:%M:%S)"
  caffeinate -i $G calibrate --checkpoint ${CK[$m]}/checkpoint --split all --dataset data/pilot_v3_calib7 > reports/phase6b/calibrate_$m.log 2>&1
  echo "EXIT calibrate_$m $?"
done
for m in a2 b2 a4; do
  cal=$(.venv/bin/python -c "import json,sys;t=open('reports/phase6b/calibrate_$m.log').read();print(json.loads(t[t.index('{'):])['path'])")
  echo "== final $m $(date -u +%H:%M:%S) $cal"
  caffeinate -i $G evaluate --checkpoint ${CK[$m]}/checkpoint --split all --dataset data/pilot_v3_final8 --calibration $cal --baselines > reports/phase6b/eval_final_$m.log 2>&1
  echo "EXIT final_$m $?"
done
echo "== profile $(date -u +%H:%M:%S)"
caffeinate -i .venv/bin/python scripts/profile_lora_step.py configs/e4b_text.yaml data/pilot_v3 reports/phase6b/lora_step_e4b_recompute.json 6 --recompute > reports/phase6b/lora_step_e4b_recompute.log 2>&1
echo "EXIT profile $?"
echo "ALL DONE $(date -u +%H:%M:%S)"
