import os
import sys
import torch
import torch.nn.functional as F
import pickle
import random
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.set_printoptions(threshold = sys.maxsize)
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM

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


if __name__ == "__main__" : 
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="../data/", help="Dataset Path")
    ap.add_argument("--data_name", default="AFG", help="Dataset Name")
    ap.add_argument("--model_path", default="./roberta_finetuned/My_Path/", help="Model Path")

    args = ap.parse_args()
    
    # Load Finetuned RoBERTa-base Tokenizer and Model
    llm_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast = True)
    llm_path = str(args.model_path) + str(args.data_name)
    llm_model = AutoModelForMaskedLM.from_pretrained(llm_path)
    llm_model.to("cuda")
    print(llm_path)
    text_path = str(args.data_path) + str(args.data_name) + "/quintuple.h5"
    df_quintuple = pd.read_hdf(text_path)
    df_text = pd.DataFrame(df_quintuple.iloc[:, 4])
    print(text_path, df_text.shape)
    
    num_sentences = df_text.shape[0]
    sentence_indices = np.arange(num_sentences)
    all_embeddings = torch.zeros((num_sentences, 50265), dtype = torch.float)
    print(all_embeddings.shape)

    ########################################################
    # sentence_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ########################################################
    
    for i in tqdm(sentence_indices) : 
        # print(i)
        curr_sentence = df_text.iloc[i].to_numpy().tolist()
        # print(type(curr_sentence[0]), len(curr_sentence[0]))
        # print(curr_sentence[0])
        curr_embedding = forward_llm(curr_sentence, llm_tokenizer, llm_model)
        # print(curr_embedding.shape)
        all_embeddings[i, :] = curr_embedding.detach().cpu()
    # print(all_embeddings[:3, :3])
    tensor_save_path = str(args.data_path) + str(args.data_name) + "/llm_sentence_embeddings.pt"
    # x = torch.load(tensor_save_path)
    # print(type(x), x.shape)
    # print(all_embeddings[:3, :3])
    torch.save(all_embeddings, tensor_save_path)
    print("LLM Forward Complete!")