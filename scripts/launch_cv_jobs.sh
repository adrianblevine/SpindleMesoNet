#!/bin/bash
#SBATCH --job-name run_mesothelioma
#.SBATCH --output log/%j.out
#.SBATCH --error log/%j.out
#SBATCH --workdir /projects/pathology_char/pathology_char_results/mesothelioma/logs/training
#SBATCH -o slurm.%N.%j.out       # STDOUT
#SBATCH -e slurm.%N.%j.err       # STDERR

#SBATCH --mail-type=FAIL         # notifications for job done & fail
#SBATCH --mail-user=levine.adrian.b@gmail.com # send-to address

#SBATCH -p dgxV100
#.SBATCH -p gpu2080
#.SBATCH -p rtx5000

#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 8G
#SBATCH --cpus-per-task 6

N_CPU=6

RUN_ID=10-23-20-jit

SCRIPT_DIR=/projects/pathology_char/pathology_char_results/mesothelioma/scripts

#logfile=`\basename $0`"_"`date "+%Y-%m-%d_%H_%M_%S"`.log
logfile=`\basename $0`"_"$RUN_ID"_"`date "+%H%M"`.log

exec >> $logfile 2>&1

. $CONDA_ROOT/projects/alevine_prj/anaconda3/etc/profile.d/conda.sh
conda activate dgx

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo "SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
fi

echo; echo "=============================================================="
echo; echo $RUN_ID

python $SCRIPT_DIR/main_run.py --model_type=resnet18 --run_id=$RUN_ID --tumor_dir=images_40x/tumor_annotated_1 --benign_dir=images_40x/benign --n_epochs=15 --n_cpu=$N_CPU --redo_split --cross_val=5 --color_jitter  

# loop to keep slide prediction running until all slides are finished
echo; echo "=============================================================="
for i in 1 2 3 4 5
do
    echo; echo; echo "CV RUN "$i
    EXIT_CODE=1
    while [ $EXIT_CODE -gt 0 ]
    do
        python $SCRIPT_DIR/prediction.py --n_workers=$N_CPU --run_id=$RUN_ID --cv_idx=$i 
        EXIT_CODE=$?
    done

    # basic slide level prediction
    python $SCRIPT_DIR/slide_prediction.py --run_id=$RUN_ID --cv_idx=$i
done


# at the end do RNN level prediction
echo; echo "=============================================================="
for i in 1 2 3 4 5
do
    echo; echo; echo "CV RUN "$i

    python $SCRIPT_DIR/rnn_prediction.py --n_process=$N_CPU --preprocess_data --cv_idx=$i --run_id=$RUN_ID --train --predict
done






