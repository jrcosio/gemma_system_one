#!/bin/zsh
# Fase 6: evaluaciones del protocolo (reports/phase6-protocol.md), en serie. Salida: reports/phase6/eval_*.log
set -u
A2=runs/pilot_ce_v3/20260923T005721Z; A2C=$A2/calibration/calibration-20260923T043407Z.json
B2=runs/pilot_lora_v3/20260923T012208Z; B2C=$B2/calibration/calibration-20260923T043504Z.json
A4=runs/e4b_experiment/20260923T204945Z; A4C=$A4/calibration/calibration-20260923T210339Z.json
G=.venv/bin/gso
run() { local name=$1; shift; echo "== $name $(date -u +%H:%M:%S)"; caffeinate -i $G evaluate "$@" > reports/phase6/eval_$name.log 2>&1; echo "EXIT $name $?"; }
run val_a2 --checkpoint $A2/checkpoint --split validation --calibration $A2C
run val_b2 --checkpoint $B2/checkpoint --split validation --calibration $B2C
run val_a4 --checkpoint $A4/checkpoint --split validation --calibration $A4C
run holdout_a2 --checkpoint $A2/checkpoint --split all --dataset data/pilot_v3_holdout6_clean --calibration $A2C --baselines
run holdout_b2 --checkpoint $B2/checkpoint --split all --dataset data/pilot_v3_holdout6_clean --calibration $B2C --baselines
run holdout_a4 --checkpoint $A4/checkpoint --split all --dataset data/pilot_v3_holdout6_clean --calibration $A4C --baselines
run faultK_b2 --checkpoint $B2/checkpoint --split all --dataset data/pilot_v3_holdout6_faultK --calibration $B2C
run faultK8_b2 --checkpoint $B2/checkpoint --split all --dataset data/pilot_v3_holdout6_faultK8 --calibration $B2C
run faultK_a4 --checkpoint $A4/checkpoint --split all --dataset data/pilot_v3_holdout6_faultK --calibration $A4C
run faultK8_a4 --checkpoint $A4/checkpoint --split all --dataset data/pilot_v3_holdout6_faultK8 --calibration $A4C
echo "ALL DONE $(date -u +%H:%M:%S)"
