import os
import random
import argparse

from tqdm import tqdm

def load_corpus(path):
    dids = set()
    for line in tqdm(open(path, encoding='utf-8'), desc='Loading corpus...'):
        did = line.split('\t')[0]
        dids.add(did)
    print('Loading corpus done. doc num:{}...'.format(len(dids)))
    return dids

def load_queries(path):
    queries = {}
    for line in tqdm(open(path, encoding='utf-8'), desc='Loading queries...'):
        qid, query_text = line.split('\t')
        queries[qid] = query_text
    print('Loading queries done. query num:{}...'.format(len(queries)))
    return queries

def load_qrel(path):
    qrel = {}
    for line in tqdm(open(path, encoding='utf-8'), desc='Loading qrel...'):
        qid, _, did, label = line.split()
        if int(label) < 1:
            continue
        qrel[qid] = did
    print('Loading qrel done. query num:{}...'.format(len(qrel)))
    return qrel

def load_run_pairs(path, queries, docs, qrels):
    print('Loading run pairs ...')
    run = {}
    for line in tqdm(open(path, encoding='utf-8')):
        qid, _, did, rank, score, _ = line.split()
        if qid not in qrels or qid not in queries or did not in docs:
            print('data not exists in query set or doc set:{}'.format(line))
            continue
        if qid in run:
            run[qid].append(did)
        else:
            run[qid] = [did]
    print('Loading run pairs  done. query num:{}...'.format(len(run)))
    return run

def set_seed(seed):
    random.seed(seed)

def main(args):
    queries = load_queries(args.query_file)
    docs = load_corpus(args.corpus_file)
    qrels = load_qrel(args.qrel_file)
    run_pairs = load_run_pairs(args.run_pairs_file, queries, docs, qrels)

    set_seed(args.seed)

    cand_qids = list(run_pairs.keys())
    random.shuffle(cand_qids)
    sampled_qids = random.sample(cand_qids, args.query_num)

    # cand_dids = {qid: random.sample(run_pairs[qid], args.negtive_doc_num) for qid in sampled_qids}

    output_dir = str(args.query_num) + '-' + str(args.negtive_doc_num) + '-' + str(args.seed)
    output_dir = os.path.join(args.output_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, '{}_pairs'.format(args.mode))

    with open(output_file, 'w') as out:
        for qid in sampled_qids:
            for i, did in enumerate(run_pairs[qid]):
                if args.mode == 'dev':
                    out.write('{} {} {} {} {} {}\n'.format(qid, 'Q0', did, i+1, 'score', 'IndriQueryLikelihood'))
                else:
                    out.write('{} {}\n'.format(qid, did))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create Json-formatted file from MS MARCO dataset.')
    parser.add_argument('--corpus_file', required=True)
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--qrel_file', required=True)
    parser.add_argument('--run_pairs_file', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seed', required=True, default=42, type=int)
    parser.add_argument('--query_num', required=True, default=100, type=int)
    parser.add_argument('--negtive_doc_num', required=True, default=100, type=int)

    args = parser.parse_args()
    main(args)