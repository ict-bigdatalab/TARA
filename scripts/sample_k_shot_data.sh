#!/bin/bash

for seed in 13 21 42 87 100
do
    for mode in train dev
    do
        python src/sample_few_shot_data.py \
            --mode ${mode} \
            --corpus_file  /data/mxy/cooperative-irgan/data/msmarco-document/msmarco-docs.tsv \
            --query_file  /data/mxy/data/msmarco/original/msmarco-doc${mode}-queries.tsv \
            --qrel_file /data/mxy/data/msmarco/original/msmarco-doc${mode}-qrels.tsv \
            --run_pairs_file /data/mxy/data/msmarco/original/msmarco-doc${mode}-top100\
            --seed ${seed} \
            --query_num 20 \
            --negtive_doc_num 100 \
            --output_dir /data/mxy/data/msmarco/few-shot
    done
done