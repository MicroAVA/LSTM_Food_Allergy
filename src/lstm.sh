#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow
declare -a databases=("latent-40" "deep_forest" "lasso" "mrmr" "rfe" "raw")
declare -a methods=("lstm" "gru" "on-lstm")
for database in "${databases[@]}";do
  echo ${database}
  for method in "${methods[@]}";do
    echo  ${method}
    python tf_dynamic_rnn.py --data ${database}  --cell ${method}
  done
done
conda deactivate