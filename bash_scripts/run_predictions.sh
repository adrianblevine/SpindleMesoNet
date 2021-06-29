#!/bin/bash

# this will restart the prediction script if it exits due to any sort
# of error (i.e. memory) until the entire script finishes
# however, as far as I know, the only way to cancel it is bu exiting
# the terminal shell

RUN_ID=10-23-20-normjit
GPU=rtx5000

EXIT_CODE=1
while [ $EXIT_CODE -gt 0 ]
do
    $1
    srun -p $GPU --cpus-per-task 4 python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/prediction.py --n_workers=4 --run_id=$RUN_ID --test_set=external
    EXIT_CODE=$?
done

while [ $EXIT_CODE -gt 0 ]
do
    $1
    srun -p $GPU --cpus-per-task 4 python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/prediction.py --n_workers=4 --run_id=$RUN_ID --test_set=referrals
    EXIT_CODE=$?
done
