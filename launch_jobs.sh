#!/bin/bash
#SBATCH --job-name run_mesothelioma
#.SBATCH --output log/%j.out
#.SBATCH --error log/%j.out
#SBATCH --workdir /projects/pathology_char/pathology_char_results/mesothelioma/logs/training_logs
#SBATCH -o slurm.%N.%j.out       # STDOUT
#SBATCH -e slurm.%N.%j.err       # STDERR

#SBATCH --mail-type=FAIL         # notifications for job done & fail
#SBATCH --mail-user=levine.adrian.b@gmail.com # send-to address

#.SBATCH -p dgxV100
#.SBATCH -p gpu2080
#SBATCH -p rtx5000

#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 4G
#SBATCH --cpus-per-task 8
N_CPU=8

BASE_ID=5-29-20-40x_list1_jitter

#logfile=`\basename $0`"_"`date "+%Y-%m-%d_%H_%M_%S"`.log
logfile=`\basename $0`"_"$BASE_ID"_"`date "+%H%M%S"`.log
exec >> $logfile 2>&1

. $CONDA_ROOT/projects/alevine_prj/anaconda3/etc/profile.d/conda.sh
conda activate dgx

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo "SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
fi

for i in 1 2 3 4 5
do
    RUN_ID=$BASE_ID'_'$i
    echo; echo "=============================================================="
    echo $RUN_ID

    #python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/main_run.py --model_type=resnet18 --run_id=$RUN_ID --tumor_dir=images_10x_norm/tumor_annotated_0.5 --benign_dir=images_10x_norm/benign --n_epochs=10 --n_cpu=$N_CPU --test_list=slides/test_list_1.txt --redo_split

    python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/main_run.py --model_type=resnet18 --run_id=$RUN_ID --tumor_dir=images_40x/tumor_annotated_1 --benign_dir=images_40x/benign --n_epochs=6 --n_cpu=$N_CPU --test_list=slides/test_list_1.txt --redo_split --apply_color_jitter
      
    # loop to keep slide prediction running until all slides are finished
    EXIT_CODE=1
    while [ $EXIT_CODE -gt 0 ]
    do
        $1
        python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/prediction.py --n_workers=$N_CPU --run_id=$RUN_ID --cv_run=1 
        #--test_img_dirs=benign_VR_0.25,meso_VR_0.25
        EXIT_CODE=$?
    done
    
    # basic slide level prediction
    python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/slide_prediction.py --run_id=$RUN_ID --cv_run=1

done


# at the end do RNN level prediction
for i in 1 2 3 4 5
do
    RUN_ID=$BASE_ID'_'$i
    echo; echo "=============================================================="
    echo $RUN_ID

    python rnn_prediction.py --n_process=$N_CPU --preprocess_data --cv_run=1 --run_id=$RUN_ID --train --predict
done






#python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/main_run.py --model_type=resnet18 --run_id=4-9-20-40x
 
#python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/main_run.py --model_type=resnet18 --run_id=4-9-20-40x_jitter --apply_color_jitter
 

#python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/main_run.py --model_type=inception --run_id=4-9-20-40x_inception

#python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/main_run.py --model_type=resnet18 --run_id=4-9-20-40x_stride0.5 --tumor_dir=images_40x/tumor_annotated_0.5


#python /projects/pathology_char/pathology_char_results/mesothelioma/scripts/main_run.py --model_type=resnet18 --run_id=$RUN_ID --tumor_dir=images_40x_norm/tumor_annotated_1 --benign_dir=images_40x_norm/benign --n_epochs=6 --n_cpu=$N_CPU



