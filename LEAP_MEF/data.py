import torch
from torch.utils import data
import numpy as np
import utils
import pickle
import collections

 
# empirical distribution/counts s/r/o in one day, predict r
class DistData(data.Dataset):
      def __init__(self, path, dataset, num_nodes, num_rels, set_name):
            data, times = utils.load_quadruples(path + dataset, set_name + '.txt')
            true_prob_s, true_prob_r, true_prob_o = utils.get_true_distributions(path, data, num_nodes, num_rels, dataset, set_name)
            times = torch.from_numpy(times)
            self.len = len(times)
            if torch.cuda.is_available():
                  true_prob_s = true_prob_s.cuda()
                  true_prob_r = true_prob_r.cuda()
                  true_prob_o = true_prob_o.cuda()
                  times = times.cuda()

            self.times = times
            self.true_prob_s = true_prob_s
            self.true_prob_r = true_prob_r
            self.true_prob_o = true_prob_o

      def __len__(self):
            return self.len

      def __getitem__(self, index):
            return self.times[index], self.true_prob_s[index], self.true_prob_r[index], self.true_prob_o[index] 
