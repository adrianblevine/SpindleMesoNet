#!/bin/bash
#SBATCH --job-name run_mesothelioma
#.SBATCH --output log/%j.out
#.SBATCH --error log/%j.out
#SBATCH --workdir /projects/pathology_char/pathology_char_results/mesothelioma/logs/prediction
#SBATCH -o slurm.%N.%j.out       # STDOUT
#SBATCH -e slurm.%N.%j.err       # STDERR

#SBATCH --mail-type=FAIL         
#SBATCH --mail-type=END 
#SBATCH --mail-user=levine.adrian.b@gmail.com
 
#SBATCH -p dgxV100
#.SBATCH -p gpu2080
#.SBATCH -p rtx5000

#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 4G
#SBATCH --cpus-per-task 6

N_CPU=6

RUN_ID=9-11-20-5xcv_norm_jitter

SCRIPT_DIR=/projects/pathology_char/pathology_char_results/mesothelioma/scripts

#logfile=`\basename $0`"_"`date "+%Y-%m-%d_%H_%M_%S"`.log
logfile=`\basename $0`"."$RUN_ID.test_preds.log

exec >> $logfile 2>&1

. $CONDA_ROOT/projects/alevine_prj/anaconda3/etc/profile.d/conda.sh
conda activate dgx

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo "SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
fi

echo; echo $RUN_ID

# loop to keep slide prediction running until all slides are finished
EXIT_CODE=1
while [ $EXIT_CODE -gt 0 ]
do
  python $SCRIPT_DIR/prediction.py --n_workers=$N_CPU --run_id=$RUN_ID --test_set 
  EXIT_CODE=$?
done
