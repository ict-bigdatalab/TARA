import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from trainer import msamrco_eval

def read_qrelfile(file):
    result = {}
    for line in tqdm(open(file, 'r'), desc='loading qrels (by line)', leave=False):
        qid, _, docid, label = line.split()
        if int(label) < 1:
            continue
        result.setdefault(qid, {})[docid] = int(label)
    return result
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, help="A dictionary contains conditions that the experiment results need to fulfill (e.g., tag, task_name, few_shot_type)")
    parser.add_argument("--log_dir", type=str, default="log", help="Log path.")
    parser.add_argument("--eval_run_name", type=str, default="best-ckpt.small.dev.run", help="eval file path.")
    parser.add_argument("--qrel_file", type=str, default="qrel file", help="qrel path.")
    parser.add_argument("--metrics", type=str, default='', help="Validation metric name")
    args = parser.parse_args()

    # Parent dir: petX /data/mxy/data/msmarco/models/few-shot/10-100/bert-base/13/pet0
    result_dirs = os.listdir(args.log_dir)

    results = {}
    best_params = None
    best_result = 0
    for item in result_dirs:
        eval_run = os.path.join(args.log_dir+f'{item}', args.eval_run_name)
        if not os.path.exists(eval_run):
            continue
        output = msamrco_eval(args.qrel_file, eval_run)
        if output > best_result:
            best_result = output
            best_params = str(item)
        results[item] = output
    
    print('best-params={}\tbest-result={}'.format(best_params, best_result))

    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[0])}
    with open(os.path.join(args.log_dir, 'eval_result.text'), 'w') as fout:
        fout.write('best-params={}\tbest-result={}\n'.format(best_params, best_result))
        for params, metric in results.items():
            fout.write('params={}\tresult={}\n'.format(params, metric))


if __name__ == '__main__':
    main()