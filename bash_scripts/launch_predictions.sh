#!/bin/bash

#SBATCH --job-name meso_pred
#SBATCH --output log/%j.out
#SBATCH --error log/%j.out
#SBATCH --workdir /home/alevine/mesothelioma/logs/prediction
#SBATCH -p dgxV100
#SBATCH -o slurm.%N.%j.out       # STDOUT
#SBATCH -e slurm.%N.%j.err       # STDERR
#SBATCH --mail-type=FAIL         # notifications for job done & fail
#SBATCH --mail-user=levine.adrian.b@gmail.com # send-to address
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8


logfile=`\basename $0`"_"`date "+%Y-%m-%d_%H_%M_%S"`.log
exec >> $logfile 2>&1

. $CONDA_ROOT/projects/alevine_prj/anaconda3/etc/profile.d/conda.sh
conda activate dgx

python /home/alevine/mesothelioma/scripts/prediction.py --n_workers=8 --test_set --run_id=10-23-20-norm --output_dir=predictions_test

#python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/prediction.py --run_id=4-9-20-10x --n_workers=4
