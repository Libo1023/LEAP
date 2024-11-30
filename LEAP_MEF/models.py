import time
import math
import torch
import random
import warnings
import itertools
import collections
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")

from utils import *

class Basic_Transformer (nn.Module) :
    def __init__(self, dim_in, dim_out) :
        super(Basic_Transformer, self).__init__()
        self.out_dim = dim_out
        self.Wq = nn.Linear(in_features = dim_in, out_features = dim_out, bias = True)
        self.Wk = nn.Linear(in_features = dim_in, out_features = dim_out, bias = True)
        self.Wv = nn.Linear(in_features = dim_in, out_features = dim_out, bias = True)
        
    def forward(self, x) :
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        q_k = torch.mm(Q, K.T)      
        q_k = q_k / np.sqrt(self.out_dim)
        Q_K = F.softmax(q_k, dim = 1) 
        out = torch.mm(Q_K, V)

        out = F.relu(out)
        
        return out

# Multi-event forecasting
class leap_mef (nn.Module) : 
    def __init__(self, h_dim, num_rels, llm_dim, seq_len) : 
        super().__init__()
        self.h_dim = h_dim
        self.llm_dim = llm_dim
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy

        ###################################################################################
        # With self-attention
        self.layer_transformer = Basic_Transformer(dim_in = llm_dim, dim_out = h_dim)
        self.layer_prediction = nn.Linear(h_dim, num_rels)

        # Without-self attention
        # self.layer_prediction = nn.Linear(llm_dim, num_rels)
        ###################################################################################

    def forward(self, t_list, true_prob_r, daily_embed_dict) : 
        pred, idx, _ = self.__get_pred_embeds(t_list, daily_embed_dict)
        loss = self.criterion(pred, true_prob_r[idx])
        return loss

    def __get_pred_embeds(self, t_list, daily_embed_dict) : 
        
        sorted_t, idx = t_list.sort(0, descending=True)

        ###################################################################################################################

        if torch.cuda.is_available() : 
            global_batch_embeddings = torch.zeros((len(sorted_t), self.h_dim)).cuda()    # With self-attention
            # global_batch_embeddings = torch.zeros((len(sorted_t), self.llm_dim)).cuda()  # Without self-attention
        else : 
            global_batch_embeddings = torch.zeros((len(sorted_t), self.h_dim))
        
        for i in range(0, len(sorted_t)) : 
            curr_time_id = sorted_t[i].item()

            if curr_time_id == 0 : 
                local_batch_embeddings = torch.zeros((10, self.llm_dim))
            
            elif curr_time_id > 0 and curr_time_id < self.seq_len : 
                for j in range(0, curr_time_id) : 
                    if j == 0 : 
                        local_batch_embeddings = daily_embed_dict[0]
                        continue
                    local_batch_embeddings = torch.cat((local_batch_embeddings, daily_embed_dict[j]), dim = 0)

            elif curr_time_id == 1096 : 
                for j in range(783-self.seq_len, 783) : 
                    if j == (783-self.seq_len) : 
                        local_batch_embeddings = daily_embed_dict[j]
                        continue
                    local_batch_embeddings = torch.cat((local_batch_embeddings, daily_embed_dict[j]), dim = 0)
            
            elif curr_time_id > 1096 and curr_time_id < (1096+self.seq_len) : 
                for j in range(1096, curr_time_id) : 
                    if j == 1096 : 
                        local_batch_embeddings = daily_embed_dict[1096]
                        continue
                    local_batch_embeddings = torch.cat((local_batch_embeddings, daily_embed_dict[j]), dim = 0)

            else : 
                for j in range(curr_time_id-self.seq_len, curr_time_id) : 
                    if j == (curr_time_id-self.seq_len) : 
                        local_batch_embeddings = daily_embed_dict[j]
                        continue
                    local_batch_embeddings = torch.cat((local_batch_embeddings, daily_embed_dict[j]), dim = 0)

            if torch.cuda.is_available() : 
                local_batch_embeddings = local_batch_embeddings.cuda()
            else : 
                local_batch_embeddings = local_batch_embeddings

            #############################################################################
            # With self-attention
            out_local = self.layer_transformer(local_batch_embeddings)
            aggregated_out_local = torch.mean(out_local, dim = 0)

            # Without self-attention
            # aggregated_out_local = torch.mean(local_batch_embeddings, dim = 0)
            ##############################################################################
            
            global_batch_embeddings[i, :] = aggregated_out_local

        if torch.cuda.is_available() : 
            feature = global_batch_embeddings.cuda()
        else : 
            feature = global_batch_embeddings
        
        pred = self.layer_prediction(feature)
        #####################################################################################################################
        
        return pred, idx, feature
        
    def predict(self, t_list, true_prob_r, daily_embed_dict) : 
        pred, idx, feature = self.__get_pred_embeds(t_list, daily_embed_dict)
        if true_prob_r is not None : 
            loss = self.criterion(pred, true_prob_r[idx])
        else : 
            loss = None
        return loss, pred, feature

    def evaluate(self, t, true_prob_r, daily_embed_dict) : 
        loss, pred, _ = self.predict(t, true_prob_r, daily_embed_dict)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, 
                                          sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, 
                                          sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]
        true_prob_r = true_prob_r.view(-1)  
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss, pred

