# -*- coding: utf-8 -*-

from util import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AdamW
import numpy as np
from pydantic import BaseModel
import torch
from torch import nn
from datasets import load_dataset
import tqdm
import string
import math
import time
import dataloader
import os
import random
import copy

class SurrogateModel(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', num_labels=2):
        super(SurrogateModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

class RLAgent:
    def __init__(self, surrogate_model, tokenizer, reward_function, action_space):
        self.surrogate_model = surrogate_model
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.action_space = action_space

    def trigger_generation(self, document, max_trigger_length=5):
        trigger = []
        for _ in range(max_trigger_length):
            input_text = " ".join(trigger) + " [MASK]"
            inputs = self.tokenizer(input_text, return_tensors='pt')
            outputs = self.surrogate_model(**inputs)
            logits = outputs.logits
            next_word_id = torch.argmax(logits, dim=-1).item()
            next_word = self.tokenizer.decode(next_word_id)
            trigger.append(next_word)
        return " ".join(trigger)
    
    def word_substitution(self, document, synonyms, max_substitutions=5):
        words = document.split()
        for _ in range(max_substitutions):
            word_to_replace = random.choice(words)
            if word_to_replace in synonyms:
                synonym = random.choice(synonyms[word_to_replace])
                words = [synonym if word == word_to_replace else word for word in words]
        return " ".join(words)

    def calculate_reward(self, original_doc, perturbed_doc, query):
        original_score = self.surrogate_model(self.tokenizer(query + original_doc, return_tensors='pt')).logits
        perturbed_score = self.surrogate_model(self.tokenizer(query + perturbed_doc, return_tensors='pt')).logits
        return self.reward_function(original_score, perturbed_score)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class TriggerPolicyNetwork(PolicyNetwork):
    def __init__(self, input_dim, vocab_size):
        super(TriggerPolicyNetwork, self).__init__(input_dim, vocab_size)

class WordSubstitutionPolicyNetwork(PolicyNetwork):
    def __init__(self, input_dim, vocab_size):
        super(WordSubstitutionPolicyNetwork, self).__init__(input_dim, vocab_size)

def reward_function(original_score, perturbed_score):
    return torch.max(torch.tensor(0), original_score - perturbed_score)

def main():
    # Parameters
    dataset_path = 'path_to_dataset'
    num_labels = 2
    target_model_path = 'path_to_target_model'
    output_dir = 'output_dir'
    output_json = 'output.json'
    sim_score_threshold = 0.7
    synonym_num = 50
    perturb_ratio = 0.4
    discriminator_checkpoint = 'checkpoint'
    mode = 'train'

    # Load models and tokenizers
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    surrogate_model = SurrogateModel(pretrained_model=target_model_path, num_labels=num_labels).to('cuda')

    # Load data
    dataset = load_dataset(dataset_path)
    queries = dataset['train']['queries']
    documents = dataset['train']['documents']

    # Initialize RL agent
    rl_agent = RLAgent(surrogate_model, tokenizer, reward_function, action_space=['trigger', 'substitution'])

    # Train or evaluate model
    if mode == 'train':
        # Train surrogate model
        optimizer = AdamW(surrogate_model.parameters(), lr=3e-5)
        for epoch in range(3):
            surrogate_model.train()
            for query, document in zip(queries, documents):
                inputs = tokenizer(query + document, return_tensors='pt').to('cuda')
                labels = torch.tensor([1 if doc in relevant_docs else 0 for doc in documents]).to('cuda')
                outputs = surrogate_model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        # Save trained model
        surrogate_model.save_pretrained(output_dir)

    elif mode == 'eval':
        # Evaluate RL agent
        results = []
        for query, document in zip(queries, documents):
            trigger = rl_agent.trigger_generation(document)
            perturbed_doc = rl_agent.word_substitution(document, synonyms={'word': ['synonym1', 'synonym2']})
            reward = rl_agent.calculate_reward(document, perturbed_doc, query)
            results.append({'query': query, 'document': document, 'trigger': trigger, 'perturbed_doc': perturbed_doc, 'reward': reward})

        # Save results
        with open(os.path.join(output_dir, output_json), 'w') as f:
            json.dump(results, f)

if __name__ == "__main__":
    main()
