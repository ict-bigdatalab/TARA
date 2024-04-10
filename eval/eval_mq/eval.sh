
file="$1"
fname=`basename $file`
task=$(echo $file | tr "/" "\n" | tail -3| head -1)
prefix=$(echo $fname | tr "." "\n" | head -1)
python convert_ranklist2score.py ./$task/Fold1/test.txt $file ./$task/result/$prefix.fold1.score
python convert_ranklist2score.py ./$task/Fold2/test.txt $file ./$task/result/$prefix.fold2.score
python convert_ranklist2score.py ./$task/Fold3/test.txt $file ./$task/result/$prefix.fold3.score
python convert_ranklist2score.py ./$task/Fold4/test.txt $file ./$task/result/$prefix.fold4.score
python convert_ranklist2score.py ./$task/Fold5/test.txt $file ./$task/result/$prefix.fold5.score

perl Eval-Score-4.0.pl ./$task/Fold1/test.txt ./$task/result/$prefix.fold1.score ./$task/result/$prefix.result.fold1.detail 1
perl Eval-Score-4.0.pl ./$task/Fold2/test.txt ./$task/result/$prefix.fold2.score ./$task/result/$prefix.result.fold2.detail 1
perl Eval-Score-4.0.pl ./$task/Fold3/test.txt ./$task/result/$prefix.fold3.score ./$task/result/$prefix.result.fold3.detail 1
perl Eval-Score-4.0.pl ./$task/Fold4/test.txt ./$task/result/$prefix.fold4.score ./$task/result/$prefix.result.fold4.detail 1
perl Eval-Score-4.0.pl ./$task/Fold5/test.txt ./$task/result/$prefix.fold5.score ./$task/result/$prefix.result.fold5.detail 1

python gen_avg_result.py ./$task/result/$prefix.result ./$task/result/$prefix.avg-result.txt
