#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow
declare -a databases=("latent-40" "deep_forest" "lasso" "mrmr" "rfe" "raw")
for database in "${databases[@]}";do
  echo ${database}
  #python rf_diabimmune.py --data ${database}
  python svm_diabimmune.py --data ${database}
done
conda deactivate