#!/bin/bash


. $CONDA_ROOT/projects/alevine_prj/anaconda3/etc/profile.d/conda.sh
conda activate dgx

for run in 5-16-20-40x 5-17-20-40x_norm 5-21-20-40x_list1_norm 
do
    echo; echo "==========================================================="
    for i in 1 2 3 4 5
    do
        RUN_ID=$run'_'$i
        echo; echo $RUN_ID

        python rnn_prediction.py --cv_run=1 --run_id=$RUN_ID --predict
    done  
    python run_summary_stats.py '/home/alevine/mesothelioma/results/run_summaries/'$run'-rnn.csv'
done



