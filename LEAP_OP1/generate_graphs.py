import numpy as np
import os
import pickle
import dgl
import torch
from tqdm import tqdm
import argparse
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

def load_quadruples(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    x = data[np.where(data[:,3] == tim)].copy()
    x = np.delete(x, 3, 1)  # drops time column
    return x

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx

def get_big_graph(triples, num_nodes, num_rels):

    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    
    return g


def main(args):
    graph_dict = {}

    data_path = args.datapath
    
    train_data, train_times = load_quadruples(data_path, 'train.txt')
    val_data, val_times = load_quadruples(data_path, 'valid.txt')
    test_data, test_times = load_quadruples(data_path, 'test.txt')

    with open(os.path.join(data_path, 'stat.txt'), 'r') as f:
        line = f.readline()
        num_nodes, num_r, _ = line.strip().split("\t")
        num_nodes = int(num_nodes)
        num_r = int(num_r)
    print("num_nodes = ", num_nodes, ", num_r = ", num_r)

    with tqdm(total=len(train_times), desc="Generating graphs for training") as pbar:
        for tim in train_times:
            data = get_data_with_t(train_data, tim)
            graph_dict[tim] = (get_big_graph(data, num_nodes, num_r))
            pbar.update(1)

    with tqdm(total=len(val_times), desc="Generating graphs for validating") as pbar:
        for tim in val_times:
            data = get_data_with_t(val_data, tim)
            graph_dict[tim] = (get_big_graph(data, num_nodes, num_r))
            pbar.update(1)
        
    with tqdm(total=len(test_times), desc="Generating graphs for testing") as pbar:
        for tim in test_times:
            data = get_data_with_t(test_data, tim)
            graph_dict[tim] = (get_big_graph(data, num_nodes, num_r))
            pbar.update(1)

    with open(os.path.join(data_path, 'graph_dict.pkl'), 'wb') as fp:
        pickle.dump(graph_dict, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graphs')
    parser.add_argument("--datapath", type=str, default="../data/AFG",
                        help="dataset to generate graph")
    args = parser.parse_args()

    main(args)
