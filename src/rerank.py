import logging
import argparse
from pathlib import Path

import torch

import utils
from trainer import run_eval, msmarco_eval, set_seed
from modeling import BertRanker, RobertaRanker, RankingBERT_Train


log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


def main_cli():
    parser = argparse.ArgumentParser('BERT model re-ranking')
    # model args
    parser.add_argument('--task', default=None)
    parser.add_argument('--model_name', default='bert')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--eval_batch_size', default=20, type=int)
    parser.add_argument('--max_sequence_length', default=512, type=int)
    # data args
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_dir', type=Path, default=None, help='The data dir consists of query file, doc file and the qrel file')
    parser.add_argument('--query_file', type=Path)
    parser.add_argument('--doc_file', type=Path)
    parser.add_argument('--qrel_file', type=Path)
    parser.add_argument('--eval_trec_file', type=Path, required=True)
    parser.add_argument('--output_file', type=Path, required=True)
    parser.add_argument('--temp_dir', required=True, help='the temporary directory where the document data will be saved')
    parser.add_argument('--tokenize_doc', action="store_true", help='Whether to tokenize document by wordpiece.')

    args = parser.parse_args()

    set_seed(args.seed)

    # MODEL = BertRanker if args.model_name == 'bert' else RobertaRanker 
    # model = MODEL(args.model_path).cuda()
    model = RankingBERT_Train.from_pretrained(args.model_path).cuda()

    if args.data_dir:
        queries, docs, qrel = utils.read_data_from_dir(args.data_dir, args.temp_dir, args.max_sequence_length, tokenize=args.tokenize_doc)
    else:
        assert args.query_file.is_file() and args.doc_file.is_file() and args.qrel_file.is_file()
        queries = utils.read_queryfile(args.query_file, model.tokenizer)
        docs = utils.read_docfile(args.temp_dir, args.doc_file, args.max_sequence_length, tokenize_doc=args.tokenize_doc, tokenizer=model.tokenizer)
        qrel = utils.read_qrelfile(args.qrel_file, queries, docs)

    eval_run_dict = utils.read_eval_trec(args.eval_trec_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: {} n_gpu: {}.".format(device, n_gpu))

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    run_eval(args, model, queries, docs, eval_run_dict, args.output_file, device, desc='rerank')
    if args.qrel_file:
        msmarco_eval(args.qrel_file, args.output_file)
    docs.close()


if __name__ == '__main__':
    main_cli()
