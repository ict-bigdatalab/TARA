import os
import shelve
import random
import logging
from pathlib import Path
from collections import namedtuple
from tempfile import TemporaryDirectory

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class DocumentDatabase:
    def __init__(self, temp_dir='./'):
        self.temp_dir = TemporaryDirectory(dir=temp_dir)
        self.working_dir = Path(self.temp_dir.name)
        self.document_shelf_filepath = self.working_dir / 'shelf.db'
        self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                            flag='n', protocol=-1)
        self.doc_ids = []

    def add(self, doc_idx, document):
        self.document_shelf[str(doc_idx)] = document
        self.doc_ids.append(doc_idx)

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, item):
        return self.document_shelf[str(item)]

    def __contains__(self, item):
        if str(item) in self.document_shelf:
            return True
        else:
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
    
    def close(self):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

def read_data_from_dir(data_dir, temp_dir, max_sequence_length, tokenize_doc=False, tokenizer=None):
    query_file = os.path.join(data_dir, 'queries')
    doc_file = os.path.join(data_dir, 'docs')
    qrel_file = os.path.join(data_dir, 'qrel')

    queries = read_queryfile(query_file, tokenizer)
    docs = read_docfile(doc_file, temp_dir, max_sequence_length, tokenize_doc, tokenizer=tokenizer)
    qrel = None
    if qrel_file.is_file():
        qrel = read_qrelfile(qrel_file)
    return queries, docs, qrel

def read_queryfile(file, tokenizer):
    queries = {}
    for i, line in enumerate(tqdm(open(file, 'r'), desc='loading queryfile (by line)', leave=False)):
        cols = line.rstrip().split('\t')
        if len(cols) == 3:
            c_type, c_id, c_text = cols
            assert c_type == 'query'
        else:
            cols = line.rstrip().split()
            c_id = cols[0]
            c_text = ' '.join(cols[1:])
        queries[c_id] = tokenizer.tokenize(c_text.lower())
    
    logging.info('query num: {}'.format(len(queries)))
    return queries

def read_docfile(temp_dir, file, max_sequence_length, tokenize_doc=False, run_dict=None, tokenizer=None):
    if run_dict:
        did_scores = run_dict.values()
        dids = []
        for item in did_scores:
            dids += item.keys()
        dids = set(dids)
        logging.info('dids set num: {}'.format(len(dids)))

    docs = DocumentDatabase(temp_dir)
    for i, line in enumerate(tqdm(open(file, 'r'), desc='loading datafile (by line)', leave=False)):
        # if i > 50000:
            # break
        cols = line.rstrip().split('\t')
        if len(cols) != 5:
            tqdm.write(f'skipping line: `{line.rstrip()}`')
            continue
        _, doc_id, _, doc, label = cols
        if run_dict and doc_id not in dids:
            continue
        if doc_id in docs:
            continue
        if tokenize_doc:
            doc_toks = tokenizer.tokenize(' '.join(doc.split()[:max_sequence_length]).lower())
        else:
            doc_toks = doc.split()
        docs.add(doc_id, doc_toks)
    logging.info('doc num: {}'.format(len(docs)))
    return docs

def read_qrelfile(file, queries, docs):
    result = {}
    for line in tqdm(open(file, 'r'), desc='loading qrels (by line)', leave=False):
        qid, _, docid, label = line.split()
        if qid not in queries or docid not in docs:
            continue
        if int(label) < 1:
            continue
        result.setdefault(qid, {})[docid] = int(label)
    logging.info('qrel num: {}'.format(len(result)))
    return result

def read_eval_trec(file):
    result = {}
    for i, line in enumerate(tqdm(open(file, 'r'), desc='loading run (by line)', leave=False)):
        qid, _, docid, rank, score, _ = line.split()
        result.setdefault(qid, {})[docid] = 0
    return result

def read_train_pairs(file, queries, docs, qrel):
    qd_pairs = {}
    pos_num = 0
    qids = set()
    for line in tqdm(open(file, 'r'), desc='loading pairs (by line)', leave=False):
        qid, docid = line.split()
        if qid not in qrel or qid not in queries or docid not in docs:
            continue
        qids.add(qid)
        label = qrel.get(qid, {}).get(docid, 0)
        qd_pairs.setdefault(qid, {})[docid] = label 
        print(qd_pairs)
        input()
        if label > 0:
            pos_num += 1
    logging.info('training query num: {}, pos num retrieved in train_pairs(top-k): {}'.format(len(qd_pairs), pos_num))
    return qd_pairs, pos_num


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        # We want to truncate from the back or the doc side
        tokens_b.pop()

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids label ")

def convert_example_to_features(query_toks, doc_toks, label, max_seq_length, tokenizer):
    tokens_a_ids = tokenizer.convert_tokens_to_ids(query_toks)
    tokens_b_ids = tokenizer.convert_tokens_to_ids(doc_toks)
    
    inputs = tokenizer.prepare_for_model(tokens_a_ids,tokens_b_ids,add_special_tokens=True,truncation='only_second',max_length=max_seq_length,return_token_type_ids=True)
    input_ids =  inputs['input_ids']
    segment_ids = inputs['token_type_ids']

    assert len(input_ids) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.int)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.int)
    segment_array[:len(segment_ids)] = segment_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             label=label,
                            )
    return features

class PregeneratedDataset(Dataset):
    def __init__(self, args, queries, docs, train_pairs, qrels, epoch, tokenizer, mode='train'):
        self.epoch = epoch
        self.working_dir = None
        self.tokenizer = tokenizer
        self.strategy = args.strategy
        self.negative_num = args.negative_num
        self.seq_len = args.max_sequence_length

        num_samples = 0
        logging.info(f"Loading {mode} examples for epoch {epoch}")
        examples = []

        if mode == 'train':
            qids = list(train_pairs.keys())
            random.shuffle(qids)
            for qid in tqdm(qids):
                if qid not in queries:
                    continue

                pos_ids = [did for did, label in qrels[qid].items() if int(label) > 0]
                pos_ids_lookup = set(pos_ids)
                if len(pos_ids) == 0:
                    continue
                
                neg_ids = [did for did in train_pairs[qid] if did in docs and did not in pos_ids_lookup]
                if len(neg_ids) < self.negative_num:
                    continue
                
                if self.strategy == 'q':
                    pos_id = np.random.choice(pos_ids)
                    examples.append((qid, pos_id, qrels[qid][pos_id]))
                    replacement = False if len(neg_ids) > self.negative_num else True
                    neg_ids = np.random.choice(neg_ids, self.negative_num, replace=replacement)
                    examples.extend((qid, neg_id, 0) for neg_id in neg_ids)
                    num_samples += self.negative_num + 1
                elif self.strategy == 'd':
                    for pos_id in pos_ids:
                        examples.append((qid, pos_id, qrels[qid][pos_id]))
                        replacement = False if len(neg_ids) > self.negative_num else True
                        neg_ids = np.random.choice(neg_ids, self.negative_num, replace=replacement)
                        examples.extend((qid, neg_id, 0) for neg_id in neg_ids)
                        num_samples += self.negative_num + 1
                else:
                    raise ValueError('invalid strategy to sample data!')
        logging.info('Num samples for epoch-{}: {}'.format(epoch, num_samples))
        
        self.temp_dir = TemporaryDirectory(dir=args.temp_dir)
        self.working_dir = Path(self.temp_dir.name)
        input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, self.seq_len))
        input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                shape=(num_samples, self.seq_len), mode='w+', dtype=np.int32)
        segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                shape=(num_samples, self.seq_len), mode='w+', dtype=np.int32)
        labels = np.memmap(filename=self.working_dir/'labels.memmap',
                                shape=(num_samples), mode='w+', dtype=np.bool)

        instance_index = 0
        for i, example in enumerate(tqdm(examples, desc=f"epoch: {epoch}, Converting {mode} examples to features.", leave=False)): # total=num_samples, 
            qid, did, label = example
            if instance_index+1 > num_samples:
                break
            features = convert_example_to_features(queries[qid], docs[did], label, self.seq_len, tokenizer)
            labels[instance_index] = features.label
            input_ids[instance_index] = features.input_ids
            segment_ids[instance_index] = features.segment_ids
            input_masks[instance_index] = features.input_mask
            instance_index += 1
            # if i < 3:
            #     logging.info("*** Example {} ***".format(i))
            #     logging.info("query tokens: %s" % " ".join(queries[qid]))
            #     logging.info("doc tokens: %s" % " ".join(docs[did]))
            #     logging.info("segment_ids: %s" % " ".join([str(x) for x in features.segment_ids]))
            #     logging.info("label: %s" % (str(features.label)))
        # assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info('Real num samples:{}'.format(instance_index))
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.labels = labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
            torch.tensor(self.input_masks[item].astype(np.int64)),
            torch.tensor(self.segment_ids[item].astype(np.int64)),
            torch.tensor(int(self.labels[item]))
            )
        

def iter_valid_records(args, model, queries, docs, run):
    tokenizer = model.module.get_tokenizer() if hasattr(model, 'module') else model.get_tokenizer()
    batch = {'query_id': [], 'doc_id': [], 'input_ids': [], 'segment_ids': [], 'input_mask':[], 'label': []}
    for qid, did in _iter_valid_records(queries, docs, run):
        features = convert_example_to_features(queries[qid], docs[did], 0, args.max_sequence_length, tokenizer)
        batch['label'].append(features.label)
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['input_ids'].append(features.input_ids)
        batch['segment_ids'].append(features.segment_ids)
        batch['input_mask'].append(features.input_mask)
        
        if len(batch['query_id']) == args.eval_batch_size:
            batch['input_ids'] = torch.tensor(batch['input_ids']).long()
            batch['segment_ids'] = torch.tensor(batch['segment_ids']).long()
            batch['input_mask'] = torch.tensor(batch['input_mask']).float()
            batch['label'] = torch.tensor(batch['label']).long()
            yield batch
            
            batch = {'query_id': [], 'doc_id': [], 'input_ids': [], 'segment_ids': [], 'input_mask':[], 'label': []}
            
    # final batch
    if len(batch['query_id']) > 0:
        batch['input_ids'] = torch.tensor(batch['input_ids']).long()
        batch['segment_ids'] = torch.tensor(batch['segment_ids']).long()
        batch['input_mask'] = torch.tensor(batch['input_mask']).float()
        batch['label'] = torch.tensor(batch['label']).long()
        yield batch

def _iter_valid_records(queries, docs, run):
    for qid in run:
        if qid not in queries:
            continue
        # query_tok = queries[qid]
        for did in run[qid]:
            if did not in docs:
                tqdm.write(f'missing doc {did}! Skipping')
                continue
            # doc_tok = docs[did]
            yield qid, did #, query_tok, doc_tok