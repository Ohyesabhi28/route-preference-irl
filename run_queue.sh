#!/bin/bash
# Robustness experiments, queued behind any running run_ablation.py (CPU contention).
cd /c/Desktop/RL_1
while tasklist | grep -qi python && powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {\$_.CommandLine -match 'run_ablation|run_b1'} | Measure-Object | % Count" | grep -qv '^0'; do sleep 60; done
export IRL_FEATURES=v3
for n in 500 1000; do
  IRL_FAM_PATH=data/fam_$n.pkl IRL_CV_DIR=results/porto/pool$n python run_cv.py --folds 5 --seeds 0 --only-fold 0 --iters 30 --quiet > results/porto/pool/pool$n.log 2>&1
done
IRL_CV_SPLIT=taxi IRL_CV_DIR=results/porto/cv_taxi python run_cv.py --folds 5 --seeds 0 --iters 30 --quiet > results/porto/pool/taxi.log 2>&1
python run_cv.py --folds 5 --seeds 1 --iters 30 --quiet > results/porto/pool/seed1.log 2>&1
echo ALLDONE > results/porto/pool/done.flag
