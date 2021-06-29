#!/bin/bash

echo MIL-POOL
for run in 10-23-20-norm 10-23-20-jit 10-23-20-normjit
do
    for test_set in val referrals external
    do 
        echo; echo; echo $run $test_set
        python slide_prediction.py --run_id=$run --test_set=$test_set
    done
done


echo RNN
for run in 10-23-20-norm 10-23-20-jit 10-23-20-normjit
do
    echo; echo; echo VAL
    python rnn_prediction.py --run_id=$run --test_set=val --predict --evaluate --threshold=0.95
    echo; echo; echo REFERRALS
    python rnn_prediction.py --run_id=$run --test_set=referrals --predict --evaluate --threshold=0.95
    echo; echo; echo EXTERNAL
    python rnn_prediction.py --run_id=$run --test_set=external --predict --evaluate --threshold=0.8
done
