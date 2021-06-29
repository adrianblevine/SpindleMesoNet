#!/bin/bash

printf "NO COLOR JITTER:\n"

for i in 1 2 3 4 6 8 10 12 16
do
  printf "\n$i CPU:"
  
  srun -p dgxV100 --gres=gpu:1 --cpus-per-task $i python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/main_run.py --run_id=test --model_type=resnet18 --profiler_mode --n_cpu=$i
done


printf "WITH COLOR JITTER:\n"

for i in 1 2 3 4 6 8 10 12 16
do
  printf "\n$i CPU:"
  
  srun -p dgxV100 --gres=gpu:1 --cpus-per-task $i python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/main_run.py --run_id=test --model_type=resnet18 --profiler_mode --n_cpu=$i --apply_color_jitter
done
