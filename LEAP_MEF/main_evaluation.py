import os
import time
import utils
import torch
import pickle
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from data import *
from models import *

import warnings
def warn(*args, 
         **kwargs) : 
    pass
warnings.warn = warn
 
parser = argparse.ArgumentParser(description='')
parser.add_argument("--dp", type=str, default="../data/", help="data path")
parser.add_argument("--n-hidden", type=int, default=1024, help="number of hidden units")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight_decay")
parser.add_argument("-d", "--dataset", type=str, default='AFG', help="dataset to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--max-epochs", type=int, default=40, help="maximum epochs")
parser.add_argument("--seq-len", type=int, default=7)
# parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--runs", type=int, default=1, help='number of runs')
parser.add_argument("--llm_out_dim", type=int, default=50265, help="ROBERTa sentence embedding dimension")
parser.add_argument("--llm_version", type=str, default="before", help="before or after LLM fine-tuning")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
print("cuda",use_cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

###################################################################################################################
###################################################################################################################
data_path = "../data/"
data_name = args.dataset + "/"
print("Current Dataset:", data_name)

path_quad = data_path + data_name + "quadruple.txt"
path_text = data_path + data_name + "quintuple.h5"

df_quad = pd.read_csv(path_quad, sep='\t', lineterminator='\n', 
                      names=['source', 'relation', 'target', 'time'])
df_quintuple = pd.read_hdf(path_text)
df_text = pd.DataFrame(df_quintuple.iloc[:, 4])
np_data = pd.concat([df_quad, df_text], axis = 1)
np_data = np_data.to_numpy()
unique_r = np.unique(np_data[:, 1])
print("NumPy Data Shape:", np_data.shape)
print("Total Number of Unique Relations = %i" % unique_r.shape[0])

daily_embed_dict = dict()
if args.llm_version == "before" : 
    path_embeddings = "../data/" + data_name + "all_quintuple_embeddings_without_finetuning.pt"
if args.llm_version == "after" : 
    path_embeddings = "../data/" + data_name + "all_quintuple_embeddings_with_finetuning.pt"
print(path_embeddings[12:-3])
print("Loading All Prompt Embeddings......")
all_prompt_embeddings = torch.load(path_embeddings)
print("All Prompt Embeddings:", all_prompt_embeddings.shape)
curr_day_key = int(0)
curr_time_str = np_data[0, 3]
print(curr_day_key, curr_time_str)

id_s = int(0)
id_e = int(0)

for i in range(0, np_data.shape[0]) : 
    if curr_time_str == np_data[i, 3] : 
        id_e = i + 1
    else : 
        daily_embed_dict[curr_day_key] = all_prompt_embeddings[id_s:id_e, :]
        # print(daily_embed_dict[curr_day_key].shape)
        id_s = id_e
        if curr_day_key != 782 : 
            curr_day_key = curr_day_key + 1
        else : 
            curr_day_key = 1096
        curr_time_str = np_data[i, 3]
daily_embed_dict[curr_day_key] = all_prompt_embeddings[id_s:, :]
# print(daily_embed_dict[curr_day_key].shape)
# print("daily_embed_dict:", len(daily_embed_dict), daily_embed_dict.keys())
####################################################################################################################
####################################################################################################################

# eval metrics
recall_list  = []
f1_list  = []
f2_list  = []
hloss_list = []

iterations = 0 
while iterations < args.runs:
    iterations += 1
    print('****************** iterations ',iterations,)
    
    if iterations == 1:
        print("loading data...")
        num_nodes, num_rels = utils.get_total_number("../data/" + args.dataset, 'stat.txt')

        train_dataset_loader = DistData(args.dp, args.dataset, num_nodes, num_rels, set_name='train')
        valid_dataset_loader = DistData(args.dp, args.dataset, num_nodes, num_rels, set_name='valid')
        test_dataset_loader = DistData(args.dp, args.dataset, num_nodes, num_rels, set_name='test')
        
        train_loader = DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True, collate_fn=collate_4)
        # train_loader = DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=False, collate_fn=collate_4)
        valid_loader = DataLoader(valid_dataset_loader, batch_size=1, shuffle=False, collate_fn=collate_4)
        test_loader = DataLoader(test_dataset_loader, batch_size=1, shuffle=False, collate_fn=collate_4)

    ######################################################################
    model = leap_mef(h_dim = args.n_hidden, 
                     num_rels = num_rels, 
                     llm_dim = args.llm_out_dim, 
                     seq_len = args.seq_len)
    if use_cuda : model.cuda()
    ######################################################################
    
    print(args.n_hidden, "To Check Hidden Dimension")

    model_name = model.__class__.__name__
    print('Model Name: ', model_name)
    token = '{}_sequence-length{}'.format(model_name, args.seq_len)
    # token = '{}_sequence-length{}_remove-self-attention'.format(model_name, args.seq_len)
    print('Token:', token, args.dataset)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Number of Trainable Parameters: ', total_params)

    os.makedirs('./models', exist_ok=True)
    os.makedirs('./models/' + args.dataset, exist_ok=True)
    model_state_file = './models/{}/{}_{}.pth'.format(args.dataset, path_embeddings[12:-3], token)
    model_graph_file = './models/{}/{}_{}_graph.pth'.format(args.dataset, path_embeddings[12:-3], token)
    outf = './models/{}/{}_{}_test_result.txt'.format(args.dataset, path_embeddings[12:-3], token)

    @torch.no_grad()
    def evaluate(data_loader, dataset_loader, set_name='valid'):
        model.eval()
        true_rank_l = []
        prob_rank_l = []
        total_loss = 0
        for i, batch in enumerate(tqdm(data_loader)):
            batch_data, true_s, true_r, true_o = batch
            batch_data = torch.stack(batch_data, dim=0)
            # print("main: ", batch_data)
            true_r = torch.stack(true_r, dim=0)
            true_s = torch.stack(true_s, dim=0)
            true_o = torch.stack(true_o, dim=0)
            true_rank, prob_rank, loss, _ = model.evaluate(batch_data, true_r, daily_embed_dict)
            true_rank_l.append(true_rank.cpu().tolist())
            prob_rank_l.append(prob_rank.cpu().tolist())
            total_loss += loss.item()
    
        print('{} results'.format(set_name)) 
        hloss, recall, f1, f2 = utils.print_eval_metrics(true_rank_l,prob_rank_l)
        reduced_loss = total_loss / (dataset_loader.len / 1.0)
        print("{} Loss: {:.6f}".format(set_name, reduced_loss))
        return hloss, recall, f1, f2


    def train(data_loader, dataset_loader):
        model.train()
        total_loss = 0
        t0 = time.time()
        for i, batch in enumerate(tqdm(data_loader)):
            batch_data, true_s, true_r, true_o = batch
            batch_data = torch.stack(batch_data, dim=0)
            # print("main: ", batch_data.shape, batch_data)
            true_r = torch.stack(true_r, dim=0)
            true_s = torch.stack(true_s, dim=0)
            true_o = torch.stack(true_o, dim=0)
            loss = model(batch_data, true_r, daily_embed_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        t2 = time.time()
        reduced_loss = total_loss / (dataset_loader.len / args.batch_size)
        print("Epoch {:04d} | Loss {:.6f} | time {:.2f} {}".format(
            epoch, reduced_loss, t2 - t0, time.ctime()))
        return reduced_loss

 

    bad_counter = 0
    # loss_small =  float("inf")
    valid_recall_best = 0.0
    try:
        print("start training...")
        for epoch in range(1, args.max_epochs+1):
            train_loss = train(train_loader, train_dataset_loader)
            valid_loss, recall, f1, f2 = evaluate(valid_loader, valid_dataset_loader, set_name='Valid')

            # if valid_loss < loss_small:
            if recall > valid_recall_best : 
                valid_recall_best = recall
                # loss_small = valid_loss
                bad_counter = 0
                print('save better model...')
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'global_emb': None}, model_state_file)
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
        print("training done")

    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early, epoch', epoch)

    # Load the best saved model.
    print("\nstart testing...")
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    hloss, recall, f1, f2 = evaluate(test_loader, test_dataset_loader, set_name='Test')
    print(args)
    print(token, args.dataset)
    recall_list.append(recall)
    f1_list.append(f1)
    f2_list.append(f2)
    hloss_list.append(hloss)


print('finish training, results ....')
# save average results
recall_list = np.array(recall_list)
f1_list = np.array(f1_list)
f2_list = np.array(f2_list)
hloss_list = np.array(hloss_list)

recall_avg, recall_std = recall_list.mean(0), recall_list.std(0)
f1_avg, f1_std = f1_list.mean(0), f1_list.std(0)
f2_avg, f2_std = f2_list.mean(0), f2_list.std(0)
hloss_avg, hloss_std = hloss_list.mean(0), hloss_list.std(0)

print('--------------------')
print("Rec  weighted: {:.4f}".format(recall_avg))
print("F1   weighted: {:.4f}".format(f1_avg))
beta=2
print("F{}  weighted: {:.4f}".format(beta, f2_avg))
print("hamming loss: {:.4f}".format(hloss_avg))

# save it !!! 
all_results = [
    recall_list, 
    f1_list, f2_list, hloss_list,
    [recall_avg, recall_std ],
    [f1_avg, f1_std],
    [f2_avg, f2_std],
    [hloss_avg, hloss_std]
]

save_results = [recall_avg, f1_avg, f2_avg]
with open(outf, "w") as f:
    for each_result in save_results : 
        f.write(str(each_result))
        f.write("\n")

print("Summarize results over all runs")
print("Recall List: ", recall_list)
print("F1 Score List: ", f1_list)