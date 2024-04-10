import os
import math
import shutil
import string
import logging
import random
import argparse
import subprocess
from queue import Queue
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, RandomSampler

import utils
from utils import PregeneratedDataset
from modeling import BertRanker, RobertaRanker

from transformers import AdamW, get_linear_schedule_with_warmup


torch.cuda.empty_cache()
log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

total_loss = 0.
global_step = 0
best_eval_step = 0
best_eval_epoch = 0
best_eval_score = None
stop_early_flag = False


class RandomPairSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, negtive=1):
        self.data_source = data_source
        self.negtive = negtive
        if (len(self.data_source)%(self.negtive+1)) !=0:
            raise ValueError('data length {} % {} !=0, can not pair data!'.format(len(self.data_source), self.negtive+1))

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        indices = torch.arange(len(self.data_source))
        paired_indices = indices.unfold(0, self.negtive+1, self.negtive+1)
        paired_indices = torch.stack([paired_indices[i] for i in range(len(paired_indices))])
        paired_indices = paired_indices[torch.randperm(len(paired_indices))]
        indices = paired_indices.view(-1)
        return iter(indices.tolist())

    def __len__(self):
        return len(self.data_source)


def train(args, model, queries, docs, qrel, train_pairs, valid_run):
    global best_eval_score, best_eval_epoch, stop_early_flag, best_eval_step

    tokenizer = model.get_tokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: {}, n_gpu: {}.".format(device, n_gpu))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.warm_up > 0:
        if args.strategy == 'q':
            num_train_optimization_steps = math.ceil(args.epochs * (args.negative_num+1) * args.queries_per_epoch / args.train_batch_size / args.gradient_accumulation_steps)
        elif args.strategy == 'd':
            num_train_optimization_steps = math.ceil(args.epochs * (args.negative_num+1) * args.pos_num / args.train_batch_size / args.gradient_accumulation_steps)
        logging.info('num_train_optimization_steps: {}'.format(num_train_optimization_steps))

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        if args.warm_up < 1:
            warm_up_steps = num_train_optimization_steps * args.warm_up
        else:
            warm_up_steps = args.warm_up
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, num_train_optimization_steps)
        logging.info('bert lr:{}'.format(args.learning_rate))
    else:
        # no warmup
        LR = args.learning_rate
        params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
        non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
        bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': args.learning_rate}
        optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    eval_result_file = open(os.path.join(args.model_out_dir, 'eval.txt'),'a+')
    for epoch in range(args.epochs):
        if stop_early_flag:
            break

        epoch_train_dataset = PregeneratedDataset(args, queries, docs, train_pairs, qrel, epoch, tokenizer, mode='train')
        logging.info('len dataset:{}'.format(len(epoch_train_dataset)))
        train_sampler = RandomPairSampler(epoch_train_dataset, args.negative_num)
        train_dataloader = DataLoader(epoch_train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

        loss = train_epoch(args, model, train_dataloader, queries, docs, valid_run, optimizer, scheduler, epoch, device, n_gpu)
        logging.info(f'train epoch={epoch} loss={loss}')

        if args.eval_after_epoch > 0 and (epoch+1) % args.eval_after_epoch == 0:
            eval_score = validate(args, model, queries, docs, valid_run, args.qrel_file, epoch, device)
            eval_result_file.write('epoch{}={}\n'.format(epoch, eval_score))
            logging.info(f'eval epoch={epoch}, eval score={eval_score}')
            if best_eval_score is None or eval_score > best_eval_score:
                temp_output_dir = os.path.join(args.model_out_dir, 'best-ckpt')
                if os.path.isdir(temp_output_dir):
                    shutil.rmtree(temp_output_dir)
                best_eval_epoch = epoch
                best_eval_step = global_step
                best_eval_score = eval_score
                logging.info('new top validation score {} at epoch-{}, step-{}, saving weights'.format(best_eval_score, best_eval_epoch, best_eval_step))
                model.module.save(temp_output_dir) if hasattr(model, 'module') else model.save(temp_output_dir)
        
        if args.save_model_every_epoch:
            logging.info("** ** * Saving fine-tuned model at epoch-{}** ** * ".format(epoch))
            temp_output_dir = os.path.join(args.model_out_dir, "epoch-{}".format(epoch))
            model.module.save(temp_output_dir) if hasattr(model, 'module') else model.save(temp_output_dir)

def train_epoch(args, model, train_dataloader, queries, docs, valid_run, optimizer, scheduler, epoch, device, n_gpu):
    global best_eval_step, best_eval_score, global_step, total_loss, stop_early_flag
    mean_loss = 0.
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", leave=True) as pbar:
        for i, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, labels = batch
            loss = model(input_ids, segment_ids, input_mask, labels, args.negative_num + 1)
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            args.writer.add_scalars('train', {'batch_loss': loss.item()}, global_step)
            loss.backward()
            total_loss += loss.item()
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            pbar.update(1)
            global_step += 1
            mean_loss = total_loss / global_step
            pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
            
            args.writer.add_scalars('lr', {'lr': scheduler.get_lr()[0]}, global_step)
            args.writer.add_scalars('train', {'mean_loss': mean_loss}, global_step)
            
            if args.eval_interval_steps != -1 and global_step % args.eval_interval_steps == 0:
                eval_score = validate(args, model, queries, docs, valid_run, args.qrel_file, global_step, device)
                logging.info(f'epoch={epoch}, global_step={global_step}, eval score={eval_score}')
                
                temp_output_dir = os.path.join(args.model_out_dir, "step-{}".format(global_step))
                model.module.save(temp_output_dir) if hasattr(model, 'module') else model.save(temp_output_dir)

                if best_eval_score is None or eval_score > best_eval_score:
                    temp_output_dir = os.path.join(args.model_out_dir, 'best-ckpt')
                    if os.path.isdir(temp_output_dir):
                        shutil.rmtree(temp_output_dir)
                    best_eval_step = global_step
                    best_eval_score = eval_score
                    logging.info('new top validation score {} at step-{}, saving weights'.format(best_eval_score, best_eval_step))
                    model.module.save(temp_output_dir) if hasattr(model, 'module') else model.save(temp_output_dir)
                
            if global_step >= best_eval_step + args.patience:
                stop_early_flag = True
                break

    return mean_loss

def validate(args, model, queries, docs, eval_run, qrelf, global_step, device):
    result = None
    runf = os.path.join(args.model_out_dir, f'{global_step}.run')
    run_eval(args, model, queries, docs, eval_run, runf, device)
    if args.task == 'mq2007' or args.task == 'mq2008':
        VALIDATION_METRIC = 'ndcg_cut.10'
        result = mq_eval(args.fold, runf, VALIDATION_METRIC, args.task)
    elif args.task == 'msmarco':
        result = msmarco_eval(qrelf, runf, args.task)
    else:
        VALIDATION_METRIC = 'ndcg_cut.20'
        result = trec_eval(qrelf, runf, VALIDATION_METRIC)
    args.writer.add_scalars('eval', {'eval_performance':round(result,4)}, global_step)
    return result

def run_eval(args, model, queries, docs, run, runf, device, desc='valid'):
    rerank_run = {}
    model.eval()
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        for records in utils.iter_valid_records(args, model, queries, docs, run):
            scores = model(records['input_ids'].to(device),
                            records['segment_ids'].to(device),
                            records['input_mask'].to(device),
                            None)

            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def trec_eval(qrelf, runf, metric):
    trec_eval_f = 'eval/trec_eval'
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    logging.info('Eval run file:{}, Metrics:{}'.format(runf, output))
    assert len(output) == 1
    return float(output[0].split()[2])

def mq_eval(fold_num, runf, metric, task):
    eval_f = './eval/eval_mq.sh'
    result_path = './mq_result'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    suffix = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    out_file = result_path + '/' + task + '.' + fold_num+ '.detail.'+suffix
    output = subprocess.check_output([eval_f, fold_num, runf, task, suffix]).decode().rstrip()
    with open(out_file, 'r') as f:
        data = f.readlines()
        res = []
        p_flag = False
        for line in data:
            if 'Average' not in line:
                continue
            res.append(line.strip().split()[10])
    logging.info('fold_num:{}, task:{}, p res:{}, ndcg res:{}'.format(fold_num, task, res[0], res[1]))
    if 'ndcg' in metric:
        return float(res[1])
    else:
        return float(res[0])

def msmarco_eval(qrelf, runf, task):
    # python document_ranking.py <path_to_candidate_file> <path_to_reference_file> <queries_to_exclude>
    if task == 'msmarco-document':
        eval_f = './eval/eval_msmarco_doc.sh'
    elif task == 'msmarco-passage':
        eval_f = './eval/eval_msmarco_passage.sh'
    else:
        raise ValueError('Invalid task name: {}'.format(task))

    output = subprocess.check_output([eval_f, runf, qrelf]).decode().rstrip()
    logging.info('Eval run file:{}, Metrics:{}'.format(runf, output))
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[1])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser('BERT Ranker')
    # model args
    parser.add_argument('--task', default=None)
    parser.add_argument('--model_name', default='bert')
    parser.add_argument('--model_path', default=None, required=True)
    parser.add_argument('--fold', default=None)
    # training args
    parser.add_argument('--warm_up', default=-1, type=float,
                        help='Either use warmup for learning rate shedule, -1 for not use and a float num for a specific warmup portion.')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--queries_per_epoch', default=-1, type=int, 
        help='if you specifiy the q strategy to organize training data per epoch, then you must set it since scheduler needs it.')
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--patience', default=300, type=int, help='Stop training when performance do not improve after several steps.')
    parser.add_argument('--negative_num', default=1, type=int,
                        help='Negative document num for a given positive doc, e.g. 1,2,...M.')
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument('--train_batch_size', default=20, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--eval_after_epoch', default=-1, type=int,
                        help="Validation after every training epoch is done.")
    parser.add_argument('--eval_interval_steps', default=-1, type=int, help='Validation after several steps, -1 default is not do eval.')
    parser.add_argument('--save_model_every_epoch', action="store_true", help='Whether to save model after each epoch.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_type', default='ce', type=str, 
                        help='CrossEntropy (ce) or Margin loss (margin).')
    parser.add_argument('--strategy', default='q', type=str,
                        help='Construct epoch dataset based on query, i.e. sample one pair (if negative num is 1) for each query (q); Or based on positive docs(d).')
    # data args
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_dir', type=Path, default=None, help='The data dir consists of query file, doc file and the qrel file')
    parser.add_argument('--query_file', type=Path)
    parser.add_argument('--doc_file', type=Path)
    parser.add_argument('--qrel_file', type=Path)
    parser.add_argument('--temp_dir', required=True, help='the temporary directory where the document data will be saved')
    parser.add_argument('--tokenize_doc', action="store_true", help='Whether to tokenize document by wordpiece.')
    parser.add_argument('--train_pairs_file', type=Path, required=True)
    parser.add_argument('--eval_trec_file', type=Path, required=True)
    parser.add_argument('--model_out_dir', required=True)

    args = parser.parse_args()
    
    args.writer = SummaryWriter(args.model_out_dir)
    set_seed(args.seed)

    MODEL = BertRanker if args.model_name == 'bert' else RobertaRanker 
    model = MODEL(args.model_path).cuda()

    if args.data_dir:
        queries, docs, qrel = utils.read_data_from_dir(args.data_dir, args.temp_dir, args.max_sequence_length, tokenize=args.tokenize_doc)
    else:
        assert args.query_file.is_file() and args.doc_file.is_file() and args.qrel_file.is_file()
        queries = utils.read_queryfile(args.query_file, model.get_tokenizer())
        docs = utils.read_docfile(args.temp_dir, args.doc_file, args.max_sequence_length, tokenize_doc=args.tokenize_doc, tokenizer=model.get_tokenizer())
        qrel = utils.read_qrelfile(args.qrel_file, queries, docs)

    train_pairs, args.pos_num = utils.read_train_pairs(args.train_pairs_file, queries, docs, qrel)
    valid_run = utils.read_eval_trec(args.eval_trec_file)

    os.makedirs(args.model_out_dir, exist_ok=True)

    train(args, model, queries, docs, qrel, train_pairs, valid_run)

    docs.close()


if __name__ == '__main__':
    main()