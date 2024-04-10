#!/usr/bin/env bash
# Setting for the new UTF-8 terminal support in Lion
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

work_path=$(dirname $(readlink -f $0))

RESULT_PATH='mq_result'
if [ ! -d "$RESULT_PATH" ];
then
mkdir $RESULT_PATH
fi

fold="$1" # which fold: 1, 2, 3, 4 or 5
input_file="$2" # prediction file
task="$3" # TASK name: mq2007 or mq2008
file_suffix="$4"

# convert to formatted file
python $work_path/eval_mq/convert_ranklist2score.py $work_path/eval_mq/$task/F${fold}.test.txt $input_file ./$RESULT_PATH/$task.$fold.score.$file_suffix

# get result according to ground truth
perl $work_path/eval_mq/Eval-Score-4.0.pl $work_path/eval_mq/$task/F${fold}.test.txt ./$RESULT_PATH/$task.$fold.score.$file_suffix ./$RESULT_PATH/$task.$fold.detail.$file_suffix 1

# print result
python $work_path/gen_avg_result.py $fold $task ./$RESULT_PATH/$task.$fold.detail.$file_suffix

# rm ./$RESULT_PATH/$task.$fold.score.$file_suffix
# rm ./$RESULT_PATH/$task.$fold.detail.$file_suffix
