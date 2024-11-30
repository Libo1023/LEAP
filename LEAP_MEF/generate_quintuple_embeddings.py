import os
import sys
import nltk
import math
import glob
import time
import torch
import pickle
import random
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter

gpu_id = int(3)
dataset_name = sys.argv[1]
llm_version = "before"

"""
From 2012-02-22 to 2013-01-01 (only in training)
Time IDs for Training   [0, 782], [1096, 1795]
Time IDs for Validation [1796, 2019]
Time IDs for Testing    [2020, 2243]
"""

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

from transformers import logging
from transformers import pipeline
from transformers import AutoTokenizer
logging.set_verbosity_error()

from datasets import Dataset
from datasets import DatasetDict
from transformers import Trainer
from transformers import LongformerModel
from transformers import TrainingArguments
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling

data_path = "../data/"
data_name = dataset_name + "/"
print("Current Dataset:", data_name)

path_quad = data_path + data_name + "quadruple.txt"
path_text = data_path + data_name + "quintuple.h5"

df_quad = pd.read_csv(path_quad, sep='\t', lineterminator='\n', 
                      names=['source', 'relation', 'target', 'time'])
df_quintuple = pd.read_hdf(path_text)
df_text = pd.DataFrame(df_quintuple.iloc[:, 4])
data = pd.concat([df_quad, df_text], axis = 1)
data = data.to_numpy()
print("Data Shape:", data.shape)
unique_r = np.unique(data[:, 1])
print("Number of Unique Relations:", unique_r.shape[0])

def build_prompt_1 (curr_data) : 
    prompt = f"""
             Subject:      {curr_data[0]}; 
             Relation:     {curr_data[1]}; 
             Object:       {curr_data[2]}; 
             Timestamp:    {curr_data[3]}; 
             Text Summary: {curr_data[4]}
             """
    return prompt

quintuple_prompts = []
for i in range(0, data.shape[0]) : 
    curr_data = data[i, :]
    curr_prompt = build_prompt_1(curr_data)
    quintuple_prompts.append(curr_prompt)
print("Total Number of Quintuples = %i" % len(quintuple_prompts))

def forward_llm (sentences, tokenizer, model) : 
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask) : 
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to("cuda")
    # Compute token embeddings
    with torch.no_grad() : 
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

llm_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
if llm_version == "before" : 
    curr_version = "FacebookAI/roberta-large"
if llm_version == "after" : 
    curr_version = "./roberta_finetuned/My_Path/" + data_name
print("LLM Version:", curr_version)
llm_model = AutoModelForMaskedLM.from_pretrained(curr_version).to("cuda")
MAX_TOKENS = llm_tokenizer.model_max_length
print("MAX_TOKENS = %i" % MAX_TOKENS)

num_quintuples = len(quintuple_prompts)
sentence_indices = np.arange(num_quintuples)
all_embeddings = torch.zeros((num_quintuples, 50265), dtype = torch.float)
print("The Shape of All Quintuple Embeddings:", all_embeddings.shape)

for i in tqdm(sentence_indices) : 
    curr_sentence = quintuple_prompts[i]
    curr_embedding = forward_llm(curr_sentence, llm_tokenizer, llm_model)
    all_embeddings[i, :] = curr_embedding

if llm_version == "before" : 
    tensor_save_path = "../data/" + data_name + "all_quintuple_embeddings_without_finetuning.pt"
if llm_version == "after" : 
    tensor_save_path = "../data/" + data_name + "all_quintuple_embeddings_with_finetuning.pt"
print("Path to Save Quintuple Prompt Embeddings:", tensor_save_path)
torch.save(all_embeddings, tensor_save_path)
x = torch.load(tensor_save_path)
print(type(x), x.shape)
print("LLM Forward Complete!")