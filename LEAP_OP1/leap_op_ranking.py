import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import ConvTransE
from aggregator import Aggregator


class leap_op_rank (nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels,
                 hyper_adj_ent, hyper_adj_rel, n_layers_hypergraph_ent, n_layers_hypergraph_rel, k_contexts,
                 h_dim, sequence_len, num_bases=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, layer_norm=False,
                 input_dropout=0, hidden_dropout=0, feat_dropout=0, use_cuda=False,
                 gpu=0):
        super(leap_op_rank, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.hyper_adj_ent = hyper_adj_ent
        self.hyper_adj_rel = hyper_adj_rel
        self.n_layers_hypergraph_ent = n_layers_hypergraph_ent
        self.n_layers_hypergraph_rel = n_layers_hypergraph_rel
        self.k_contexts = k_contexts
        self.num_ents_dis = num_ents * k_contexts
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.relation_evolve = False
        self.emb_rel = None
        self.gpu = gpu
        
        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.k_contexts, self.num_rels * 2, self.h_dim),
                                          requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)
        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(self.k_contexts, self.num_ents, self.h_dim),
                                              requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        self.loss_e = torch.nn.CrossEntropyLoss()
        self.aggregators = torch.nn.ModuleList()  # one rgcn for each context
        for contextid in range(self.k_contexts):
            self.aggregators.append(Aggregator(
                h_dim,
                num_ents,
                num_rels * 2,
                num_bases,
                num_hidden_layers,
                encoder_name,
                self_loop=self_loop,
                dropout=dropout, 
                use_cuda=use_cuda))

        self.dropout = nn.Dropout(dropout)

        # print("******************************************************************")
        # print("Aggregators Information", "\n")
        # print(len(self.aggregators), type(self.aggregators[0]))
        # print(self.aggregators[0])
        # print("******************************************************************", "\n")

        self.time_gate_weights = torch.nn.ParameterList()
        self.time_gate_biases = torch.nn.ParameterList()
        for contextid in range(self.k_contexts):
            self.time_gate_weights.append(nn.Parameter(torch.Tensor(h_dim, h_dim)))
            nn.init.xavier_uniform_(self.time_gate_weights[contextid], gain=nn.init.calculate_gain('relu'))
            self.time_gate_biases.append(nn.Parameter(torch.Tensor(h_dim)))
            nn.init.zeros_(self.time_gate_biases[contextid])

        # GRU cell for relation evolving
        self.relation_gru_cells = torch.nn.ModuleList()
        for contextid in range(self.k_contexts):
            self.relation_gru_cells.append(nn.GRUCell(self.h_dim * 2, self.h_dim))
        # The number of expected features in the input x; The number of features in the hidden state h

        # print("************************************************************************")
        # print("GRU Information", "\n")
        # print(len(self.relation_gru_cells), type(self.relation_gru_cells[0]))
        # print(self.relation_gru_cells[0])
        # print("************************************************************************", "\n")

        # decoder
        if decoder_name == "convtranse":
            self.decoders = torch.nn.ModuleList()
            for contextid in range(self.k_contexts):
                self.decoders.append(ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout))
            # print("*******************************************************************************************")
            # print("Decoder Information", "\n")
            # print(len(self.decoders), type(self.decoders[0]))
            # print(input_dropout, hidden_dropout, feat_dropout)
            # print(self.decoders[0])
            # print("*******************************************************************************************", "\n")
        else:
            raise NotImplementedError

    def get_embs(self, g_list, use_cuda):

        # dynamic_emb entity embedding matrix H is global, but is normalized before every forward
        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]

        # print("self.h shape = ", self.h.shape)               # 5 * 2594 * 200
        # print("g_list info:", len(g_list), type(g_list[0]))  # length = 3, tuple

        # print(type(g_list[0]), type(g_list[0][0]), type(g_list[0][1]))
        #       tuple            Graph Class         Graph Class 

        ent_emb_each, rel_emb_each = [], []

        # print(type(g_list), len(g_list), type(g_list[0]))
        # print(g_list[0])

        for contextid in range(self.k_contexts):
            ent_emb_context = self.h[contextid, :, :]
            rel_emb_context = self.emb_rel[contextid, :, :]
            for timid, g_each in enumerate(g_list):
                
                # g = g_each[contextid].to(self.gpu)
                g = g_each.to(self.gpu)

                # print(type(g))

                # print(type(g.r_len),  type(g.r_to_e), type(g.uniq_r))
                #       list of tuples  list of ints    1d np array
                # print(len(g.r_len), g.r_len, "\n")
                # print(len(g.r_to_e), g.r_to_e, "\n")
                # print(len(g.uniq_r), g.uniq_r, "\n")
                
                if len(g.r_len) == 0:
                    continue

                temp_e = ent_emb_context[g.r_to_e]
                
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(
                    self.num_rels * 2, self.h_dim).float()

                #######################################
                # print(type(g.r_len),  type(g.uniq_r))
                #       list of tuples, 1d np array
                # print(g.r_len, g.uniq_r)
                #######################################
                
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x = temp_e[span[0]:span[1], :]  # all entities related to a relation
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    x_input[r_idx] = x_mean

                ##########################################################
                # print(rel_emb_context.shape, x_input.shape)
                #       450 * 200              450 * 200
                x_input = torch.cat((rel_emb_context, x_input), dim=1)
                # print(x_input.shape)
                #       450 * 400 
                ##########################################################
                
                rel_emb_context = self.relation_gru_cells[contextid](x_input, rel_emb_context)  # new input hidden = h'
                rel_emb_context = F.normalize(rel_emb_context) if self.layer_norm else rel_emb_context

                ##############################
                # print(rel_emb_context.shape)
                #       450 * 200 
                ##############################

                curr_ent_emb_context = self.aggregators[contextid].forward(g, ent_emb_context,
                                                                           rel_emb_context)  # aggregated node embedding
                curr_ent_emb_context = F.normalize(curr_ent_emb_context) if self.layer_norm else curr_ent_emb_context

                time_weight = torch.sigmoid(
                    torch.mm(ent_emb_context, self.time_gate_weights[contextid]) + self.time_gate_biases[contextid])
                ##############################
                # print(self.time_gate_weights[contextid].shape)
                #       200 * 200
                # print(time_weight.shape)
                #       2594 * 200
                # print(curr_ent_emb_context.shape)
                #       2594 * 200
                # print(ent_emb_context.shape)
                #       2594 * 200
                ##############################
                ent_emb_context = time_weight * curr_ent_emb_context + (1 - time_weight) * ent_emb_context
                # print(ent_emb_context.shape)
                #       2594 * 200
                ##############################

            ent_emb_each.append(ent_emb_context)
            rel_emb_each.append(rel_emb_context)
        ent_emb_each = torch.stack(ent_emb_each)  # k, num_ent, h_dim
        rel_emb_each = torch.stack(rel_emb_each)  # k, num_rel * 2, h_dim

        if self.hyper_adj_ent is not None: 
            # 5 * 2594 * 200
            # print(ent_emb_each.shape)
            ent_emb_each = self.forward_hyergraph_ent(ent_emb_each)
            # print(ent_emb_each.shape)
            
        if self.hyper_adj_rel is not None: 
            # 5 * 450 * 200
            # print(rel_emb_each.shape)
            rel_emb_each = self.forward_hyergraph_rel(rel_emb_each)
            # print(rel_emb_each.shape)

        # print(ent_emb_each.shape, rel_emb_each.shape)
        # 5 * 2594 * 200, 5 * 450 * 200, for each time stamp
        
        return ent_emb_each, rel_emb_each

    ####################################################################################################
    # LightGCN for Entity Embeddings (1 layer)
    def forward_hyergraph_ent(self, node_repr):
        node_repr = node_repr.transpose(0, 1).contiguous().view(-1, self.h_dim)
        for n in range(self.n_layers_hypergraph_ent):
            node_repr = F.normalize(self.dropout(self.hyper_adj_ent @ node_repr)) + node_repr
        node_repr = node_repr.view(self.num_ents, self.k_contexts, self.h_dim).transpose(0, 1)
        return node_repr

    # LightGCN for Relation Embeddings (1 layer)
    def forward_hyergraph_rel(self, rel_repr):
        rel_repr = rel_repr.transpose(0, 1).contiguous().view(-1, self.h_dim)
        for n in range(self.n_layers_hypergraph_rel):
            rel_repr = F.normalize(self.dropout(self.hyper_adj_rel @ rel_repr)) + rel_repr
        rel_repr = rel_repr.view(self.num_rels * 2, self.k_contexts, self.h_dim).transpose(0, 1)
        return rel_repr
    ####################################################################################################

    def predict(self, glist, test_triplets, use_cuda, sentence_embeddings, test_contexts = None) : 

        # print(test_triplets.shape)
            
        inverse_test_triplets = test_triplets[:, [2, 1, 0]]
        inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + self.num_rels
        all_triples = torch.cat((test_triplets, inverse_test_triplets))
        sentence_embeddings = torch.cat((sentence_embeddings, 
                                         sentence_embeddings))

        e_emb, r_emb = self.get_embs(glist, use_cuda)
        pre_emb = F.normalize(e_emb, dim=-1) if self.layer_norm else e_emb             # [k, n_ents, h_dim]

        all_score_ob = []                                                              # object score using each context, len = k_contexts
        for context in range(self.k_contexts):                                         # use sub-embeddings under the context
            pre_emb_context = pre_emb[context, :, :]                                   # [n_ents, h_dim]
            r_emb_context = r_emb[context, :, :]
            out_decoder = self.decoders[context].forward(pre_emb_context, 
                                                         r_emb_context, 
                                                         all_triples, 
                                                         # New Sentence Embeddings
                                                         sentence_embeddings)
            out_decoder = out_decoder.view(-1, self.num_ents)                          # [n_triplets, n_ents]
            all_score_ob.append(out_decoder)                                           # [n_triplets, n_ents]
        all_score_ob = torch.stack(all_score_ob, dim=1)                                # [n_triplets, k_contexts, n_ents]

        # final_score_ob = torch.sum(all_score_ob, dim = 1)
        final_score_ob = torch.squeeze(all_score_ob, dim = 1)
        # print(all_score_ob.shape, final_score_ob.shape)

        return all_triples, final_score_ob

    def forward(self, glist, test_triples, use_cuda, sentence_embeddings, test_contexts = None):
        """
        :param glist: [(g),..., (g)] len=valid_history_length
        :param test_triples: triplets to be predicted, ((s, r, o), ..., (s, r, o)) len=num_triplets
        :param use_cuda: use cuda or cpu
        :param test_contexts: [(1,0,0,0,0) onehot of contextid, ...] len=num_triplets
        :return: loss_ent
        """

        # print(len(glist), len(test_triples), len(test_contexts))
        # print(type(glist[0]), len(glist[0]), type(glist[0][0]))  # tuple with len 5 for different contexts
        # print(glist[0])
        
        all_triples, final_score_ob = self.predict(glist, test_triples, use_cuda, sentence_embeddings, test_contexts)
        loss_ent = self.loss_e(final_score_ob, all_triples[:, 2])

        # print(all_triples.shape, final_score_ob.shape)  # [2*num_triplets, 3], [2*num_triplets, all_ents]

        return loss_ent
