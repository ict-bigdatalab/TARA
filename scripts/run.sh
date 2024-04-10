#!/bin/bash

# MS MARCO Document ranking dataset, query num: 372206, valid query num (has pos doc in qrel): 367012
# MS MARCO Passage ranking dataset, query num: , valid query num (has pos doc in qrel): 367012

CUDA_VISIBLE_DEVICES=0 python src/trainer.py \
  --task clueweb \
  --model_name bert \
  --model_path /data/users/liuyuan/PROP-marco \
  --queries_per_epoch 83 \
  --warm_up 0.1 \
  --epochs 100 \
  --learning_rate 1e-5 \
  --negative_num 7 \
  --train_batch_size 24 \
  --eval_batch_size 10 \
  --gradient_accumulation_steps 1 \
  --loss_type ce \
  --strategy q \
  --eval_after_epoch 1 \
  --max_sequence_length 512 \
  --save_model_every_epoch \
  --temp_dir /data/users/liuyuan/bert-reranker/temp \
  --query_file /data/users/liuyuan/clueweb09/top200/queries.txt \
  --doc_file /data/users/liuyuan/clueweb09/top200/dataset.txt \
  --qrel_file /data/users/liuyuan/clueweb09/top200/qrels.clueweb09b.1-150.txt \
  --train_pairs_file /data/users/liuyuan/clueweb09/top200/train.pairs \
  --eval_trec_file /data/users/liuyuan/clueweb09/top200/f4.run \
  --model_out_dir tmp_clueweb/

# CUDA_VISIBLE_DEVICES=0,1 python src/trainer.py \
#   --task clueweb \
#   --model_name bert \
#   --model_path /data/users/liuyuan/PROP-marco \
#   --warm_up 0.1 \
#   --queries_per_epoch 367012 \
#   --epochs 5 \
#   --patience 300 \
#   --learning_rate 1e-5\
#   --negative_num 1 \
#   --train_batch_size 24 \
#   --eval_batch_size 10 \
#   --gradient_accumulation_steps 1 \
#   --loss_type ce \
#   --strategy q \
#   --eval_after_epoch 1 \
#   --max_sequence_length 512 \
#   --save_model_every_epoch \
#   --temp_dir ./ \
#   --query_file /data/users/liuyuan/clueweb09/top200/queries.txt \
#   --doc_file /data/users/liuyuan/msmarco-doc/dataset.txt  \
#   --qrel_file /data/users/liuyuan/msmarco-doc/qrels \
#   --train_pairs_file /data/users/liuyuan/msmarco-doc/train.pairs \
#   --eval_trec_file /data/users/liuyuan/msmarco-doc/valid.run \
#   --model_out_dir tmp_clueweb/