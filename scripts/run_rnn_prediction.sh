#!/bin/bash

RUN_ID=7-23-20-jitter_5

for i in 1 2 3
do
  echo "CV RUN "$i
  srun -c 48 --gres=gpu:1 -p dgxV100 python rnn_prediction.py --preprocess_data --run_id=$RUN_ID --n_process=48 --cv_run=$i
  #srun -c 48 python rnn_prediction.py --preprocess_data --run_id=$RUN_ID --n_process=48 --cv_run=$i
  python rnn_prediction.py --run_id=$RUN_ID --train --predict 
done
