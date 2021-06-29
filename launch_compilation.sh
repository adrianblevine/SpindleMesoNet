#!/bin/bash
#SBATCH --job-name compile_vec
#SBATCH --output log/%j.out
#SBATCH --error log/%j.out
#SBATCH --workdir /projects/pathology_char/pathology_char_results/mesothelioma/logs/prediction_logs
#.SBATCH -p dgxV100
#SBATCH -o slurm.%N.%j.out       # STDOUT
#SBATCH -e slurm.%N.%j.err       # STDERR
#SBATCH --mail-type=FAIL         # notifications for job done & fail
#SBATCH --mail-user=levine.adrian.b@gmail.com # send-to address
#.SBATCH --gres=8
#.SBATCH --cpus-per-task=8
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=4G

logfile=`\basename $0`"_"`date "+%Y-%m-%d_%H_%M_%S"`.log
exec >> $logfile 2>&1

. $CONDA_ROOT/projects/alevine_prj/anaconda3/etc/profile.d/conda.sh
conda activate dgx
python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/compile_image_feature_vectors.py
