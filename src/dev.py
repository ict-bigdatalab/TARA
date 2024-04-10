import re
from tqdm import tqdm

with open('/data/users/liuyuan/bert-reranker/pre.run', 'w', encoding='utf-8') as out:
    for line in tqdm(open('/data/users/liuyuan/PRADA/msmarco_document/data/msmarco-doc/step_80000.dev.rank.tsv', 'r', encoding='utf-8')):
        line = line.strip().split('\t')
        out.write('{} 0 D{} {} 1 run\n'.format(line[0], line[1], line[2]))