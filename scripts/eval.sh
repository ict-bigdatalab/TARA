#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python src/rerank.py \
    --task msmarco \
    --model_name bert \
    --model_path /data/users/liuyuan/PRADA_trigger/msmarco_document/models/ckpt-80000 \
    --eval_batch_size 8 \
    --query_file /data/users/liuyuan/msmarco-doc/queries.txt \
    --doc_file /data/users/liuyuan/msmarco-doc/dataset.txt  \
    --qrel_file /data/users/liuyuan/msmarco-doc/qrels \
    --eval_trec_file /data/users/liuyuan/msmarco-doc/valid.run \
    --output_file  tmp.run\
    --temp_dir ./
 