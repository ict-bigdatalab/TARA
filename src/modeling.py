import os
from re import S

import torch
from torch.nn import Softmax
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import nn
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertForMaskedLM, BertPreTrainedModel, BertModel
from transformers import RobertaForMaskedLM, RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer

class RankingBERT_Train(BertPreTrainedModel):
    def __init__(self, config):
        super(RankingBERT_Train, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.out = nn.Linear(config.hidden_size, 1)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids, token_type_ids,
                position_ids, labels=None):

        attention_mask = (input_ids != 0)

        bert_pooler_output = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)[1]

        output = self.out(self.dropout(bert_pooler_output))
        # shape = [B, 1]

        if labels is not None:

            loss_fct = nn.MarginRankingLoss(margin=1.0, reduction='mean')

            y_pos, y_neg = [], []
            for batch_index in range(len(labels)):
                label = labels[batch_index]
                if label > 0:
                    y_pos.append(output[batch_index])
                else:
                    y_neg.append(output[batch_index])
            y_pos = torch.cat(y_pos, dim=-1)
            y_neg = torch.cat(y_neg, dim=-1)
            y_true = torch.ones_like(y_pos)
            assert len(y_pos) == len(y_neg)

            loss = loss_fct(y_pos, y_neg, y_true)
            output = loss, *output
        return output


class BertRanker(torch.nn.Module):

    def __init__(self, model_name_or_path):
        super().__init__()

        self.config = BertConfig.from_pretrained(model_name_or_path)
        self.bert = BertForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.bert.module if hasattr(self.bert, 'module') else self.bert
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids, token_type_ids, input_mask, labels=None, subset_num=2):
        batch_size = input_ids.size(0)
        logits  = self.bert(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids).logits
        postive_logits = logits[:, 1]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            postive_logits = postive_logits.reshape(batch_size//subset_num, subset_num)
            pairwise_labels = torch.zeros(batch_size//subset_num, dtype=torch.long).to(logits.device)
            pairwise_loss = loss_fct(postive_logits, pairwise_labels)

            return pairwise_loss
        else:
            return postive_logits

class RobertaRanker(torch.nn.Module):
    
    def __init__(self, model_name_or_path, pattern_id=0):
        super().__init__()
        self.pattern_id = pattern_id

        self.config = RobertaConfig.from_pretrained(model_name_or_path)
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.roberta.module if hasattr(self.roberta, 'module') else self.roberta
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_tokenizer(self):
        return self.tokenizer
        
    def forward(self, input_ids, token_type_ids, input_mask, labels=None, subset_num=2):
        batch_size = input_ids.size(0)
        logits = self.roberta(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids).logits
        postive_logits = logits[:, 1]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            postive_logits = postive_logits.reshape(batch_size//subset_num, subset_num)
            pairwise_labels = torch.zeros(batch_size//subset_num, dtype=torch.long).to(postive_logits.device)
            pairwise_loss = loss_fct(postive_logits, pairwise_labels)

            return pairwise_loss
        else:
            return postive_logits

class T5EncoderRanker(torch.nn.Module):
    # TODO
    def __init__(self, model_name_or_path, pattern_id=0):
        super().__init__()
        self.pattern_id = pattern_id

        self.config = RobertaConfig.from_pretrained(model_name_or_path)
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.roberta.module if hasattr(self.roberta, 'module') else self.roberta
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def forward(self, input_ids, token_type_ids, input_mask, labels=None, subset_num=2):
        batch_size = input_ids.size(0)
        logits = self.roberta(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids).logits
        postive_logits = logits[:, 1]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            postive_logits = postive_logits.reshape(batch_size//subset_num, subset_num)
            pairwise_labels = torch.zeros(batch_size//subset_num, dtype=torch.long).to(postive_logits.device)
            pairwise_loss = loss_fct(postive_logits, pairwise_labels)

            return pairwise_loss
        else:
            return postive_logits