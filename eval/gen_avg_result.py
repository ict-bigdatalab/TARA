import os
import sys

RESULT_FILTPATH = './mq_result'

if __name__ == '__main__':
  fold = int(sys.argv[1])
  if len(sys.argv) > 2:
    TASK = sys.argv[2] 
  else:
    TASK = 'mq2007'
  if len(sys.argv) > 3:
    result_file = sys.argv[3] 
  else:
    result_file = RESULT_FILTPATH+'/'+TASK+'.%d.detail'%(fold)

  cf_res = []
  for line in open(result_file, 'r'):
    if 'Average' not in line:
      continue
    for r in line.strip().split()[1:]:
      cf_res.append(float(r))
  assert len(cf_res) == 22

  print('P METRIC:')
  for i in range(11):
    if i == 10:
      print('mean P: {}'.format(cf_res[i]))
    else:
      print('P@{}: {}'.format(i+1, cf_res[i]))

  
  print('nDCG METRIC:')
  for i in range(11, 22):
    if i == 21:
      print('mean nDCG: {}'.format(cf_res[i]))
    else:
      print('nDCG@{}: {}'.format(i-10, cf_res[i]))

