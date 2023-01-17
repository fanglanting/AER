import logging
import time
#from types import _V_co
import numpy as np
import torch
from torch.autograd import Variable
import multiprocessing as mp
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
from position import *
from numba import jit
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pylab import *
from collections import Counter
from scipy import spatial
from scipy.linalg import logm
from util_func import tfidf_vector, LCS,expand_last_dim, jaccard_distance,cosine_similarity_ngrams
from collections import Counter 
from parserpara import *

PRECISION = 5
POS_DIM_ALTER = 100
import os
args, sys_argv = get_args()
GPU = args.gpu
device = torch.device('cuda:{}'.format(GPU))

class MaskGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MaskGRUCell, self).__init__()
        self.input_size = input_size
        lb, ub = -sqrt(1/hidden_size), sqrt(1/hidden_size)
        self.in2hid_w = [self.__init(lb, ub, input_size, hidden_size) for _ in range(3)]
        self.hid2hid_w = [self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)]
        self.in2hid_b = [self.__init(lb, ub, hidden_size) for _ in range(3)]
        self.hid2hid_b = [self.__init(lb, ub, hidden_size) for _ in range(3)]
    @staticmethod
    def __init(low, upper, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1).to(device) * (upper - low) + low)
        else:
            return nn.Parameter(torch.rand(dim1, dim2).to(device)* (upper - low) + low)

    def forward(self, x, hid,mask=None):
        # print(self.in2hid_w[0],x)
        # y = torch.mm(x, self.in2hid_w[0])
        r = torch.sigmoid(torch.mm(x, self.in2hid_w[0]) + self.in2hid_b[0] +
                          torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0])
        z = torch.sigmoid(torch.mm(x, self.in2hid_w[1]) + self.in2hid_b[1] +
                          torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1])
        n = torch.tanh(torch.mm(x, self.in2hid_w[2]) + self.in2hid_b[2] +
                       torch.mul(r, (torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2])))

        if mask is not None:
            M= z.shape[1]
            mask=mask.repeat(1,M)
            z = torch.mul(z, mask)
        next_hid = torch.mul(z, n) + torch.mul((1-z), hid)
        return next_hid

class GRUModel(nn.Module):
    
    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        self.grucell = MaskGRUCell(input_num, hidden_num)
        self.out_linear = nn.Linear(hidden_num, output_num)

    def forward(self, x, hid=None, mask=None):
        if hid is None:
            hid = torch.randn(x.shape[1], self.hidden_size).to(device)
        hid_out = []
        if mask is not None:
            for i in range(len(x)-1,-1,-1):
                hid = self.grucell(x[i], hid, mask[i])
                hid_out.append(hid)
        else:
            for i in range(len(x)-1,-1,-1):
                hid = self.grucell(x[i], hid)
                hid_out.append(hid)         
        # print('hid_out',hid_out)
        hid_out = torch.stack(hid_out, dim=1).to(device)
        y = self.out_linear(hid_out)
        return y, hid

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        #output = self.layer_norm(output)
        
        return output, attn


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())


    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic

class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb

class EdgeScoreInSubgraph:
    def __init__(self, src_idx_l, e_idx_l, dst_idx_l):
        self.src_idmap = dict()
        self.tar_idmap = dict()
        srcs = set(src_idx_l)
        tars = set(dst_idx_l)
        print('set ',len(srcs), len(tars))
        self.src_idx_list = src_idx_l
        self.tar_idx_list = dst_idx_l
        self.adj_mat = np.zeros((len(srcs), len(tars)))
        self.scores = np.zeros((len(srcs), len(tars)))
        self.src_dim = len(srcs)
        self.tar_dim = len(tars)
        for s in srcs:
            self.src_idmap[s] = len(self.src_idmap)
        for t in tars:
            self.tar_idmap[t] = len(self.tar_idmap)
        # print('src_idmap', self.src_idmap)
        # print('tar_idmap', self.tar_idmap)
        for (s,t) in zip(src_idx_l, dst_idx_l):
            s_id = self.src_idmap[s]
            t_id = self.tar_idmap[t]
            self.adj_mat[s_id, t_id] += 1
        # print("adj_matrix", self.adj_mat)
        self.ComputeScoreForNonZeroElement()

    def ComputeScoreForNonZeroElement(self):
        s_mat = np.reshape(self.adj_mat.T, (self.tar_dim, 1, self.src_dim, 1))
        t_mat = np.reshape(self.adj_mat, (1, self.src_dim, 1, self.tar_dim))
        expected_mat = np.matmul(s_mat, t_mat)
        #print(expected_mat.shape)
        nz_idx = np.nonzero(self.adj_mat)
        for (s,t) in zip(nz_idx[0], nz_idx[1]):
            nz = np.where(expected_mat[t, s,:, :] > 0, 1, expected_mat[t, s,:, :])
            #print(expected_mat[t,s,:,:])
            self.scores[s,t] = 1.0 * np.multiply(self.adj_mat, nz).sum() / expected_mat[t, s,:,:].sum()        
            
    # def ComputeScoreForNonZeroElement(self):
    #     nz_idxs = np.nonzero(self.adj_mat)
    #     start_time = time.time()
    #     for (s, t) in zip(nz_idxs[0], nz_idxs[1]):
    #         # print('s,t', s, t)
    #         s_vec = self.adj_mat[s:s+1, :]
    #         t_vec = self.adj_mat[:,t:t+1]
    #         expected_mat = np.dot(t_vec, s_vec)
    #         # print(expected_mat)
    #         nz = np.nonzero(expected_mat)
    #         actual_edges = self.adj_mat[nz].sum()
    #         # print('actual edges', actual_edges)
    #         expected_edges = expected_mat.sum()
    #         self.scores[s,t] = 1.0 * actual_edges / expected_edges
    #     end_time = time.time()
    #     print('zero ', end_time-start_time)
        
    def ComputeScoresForAllEdges(self):
        edge_scores = np.zeros(len(self.src_idx_list))
        edge_id = 0
        for (s, t) in zip(self.src_idx_list, self.tar_idx_list):
            src_id = self.src_idmap[s]
            tar_id = self.tar_idmap[t]
            edge_scores[edge_id] = self.scores[src_id, tar_id]
            edge_id += 1
        return edge_scores

class RAPEncoder(nn.Module):
    def __init__(self, tf_matrix, num_layers, v_size=32883, enc_dim=2, ngh_finder=None,
     verbosity=1, cpu_cores=1,rapd_dim=32,e_feat_dim=None, logger=None,dist_dim=10, RAPs='com',RAPd='pair',step=16,mask_num=[3]):
        super(RAPEncoder, self).__init__()
        self.tf_matrix = tf_matrix
        self.RAPs = RAPs
        self.RAPd = RAPd
        self.enc_dim = enc_dim
        self.rapd_dim = rapd_dim
        # print('self.rapd_dim', self.rapd_dim)
        # if self.rapd_dim > enc_dim:
            # self.rapd_dim = enc_dim
        self.num_layers = num_layers
        self.nodetime2emb_maps = None
        self.projection = nn.Linear(1, 1)  # reserved for when the internal position encoding does not match input
        self.cpu_cores = cpu_cores
        self.ngh_finder = ngh_finder
        self.verbosity = verbosity
        self.logger = logger        
        self.vocab_size = v_size
        self.z_mean = nn.Linear(self.enc_dim, self.enc_dim)
        self.z_var = nn.Linear(self.enc_dim, self.enc_dim)
        self.rap_dict = dict()
        self.e_feat_dim = e_feat_dim
        self.step = step
        self.graph = None
        self.trainable_embedding_s = nn.Sequential(nn.Linear(in_features=self.rapd_dim, out_features=self.rapd_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.rapd_dim, out_features=self.rapd_dim))
        self.trainable_embedding_d = nn.Sequential(nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim))
        self.trainable_embedding_b = nn.Sequential(nn.Linear(in_features=self.enc_dim+self.rapd_dim, out_features=self.enc_dim+self.rapd_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.enc_dim+self.rapd_dim, out_features=self.enc_dim+self.rapd_dim))
        
        if e_feat_dim !=0:
            self.trainable_embedding_feat = nn.Sequential(nn.Linear(in_features=e_feat_dim, out_features=e_feat_dim),
                                                     nn.Tanh(),
                                                     nn.Linear(in_features=e_feat_dim, out_features=10))
            self.trainable_embedding_c = nn.Sequential(nn.Linear(in_features=self.enc_dim+self.rapd_dim+e_feat_dim, out_features=self.enc_dim+self.rapd_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.enc_dim+self.rapd_dim, out_features=self.enc_dim+self.rapd_dim))

        else:
            self.trainable_embedding_c = nn.Sequential(nn.Linear(in_features=self.enc_dim+self.rapd_dim, out_features=self.enc_dim+self.rapd_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.enc_dim+self.rapd_dim, out_features=self.enc_dim+self.rapd_dim))
        mask_num = [int(ele) for ele in mask_num]
        if len(mask_num) == 1:
            self.mask_num_src = mask_num[0]
            self.mask_num_dst = mask_num[0]
        else:
            self.mask_num_src = mask_num[0]
            self.mask_num_dst = mask_num[1]
        mask_dim = 3*self.enc_dim
        self.dist_dim = dist_dim
        self.trainable_masks = [self._mask_init(mask_dim,mask_dim) for _ in range(self.mask_num_src)]
        self.trainable_masks_dest = [self._mask_init(2*dist_dim,2*dist_dim) for _ in range(self.mask_num_dst)]
        
        output_dim = 100
        self.encoder_mask = [self._masks_cnn_init(mask_dim,output_dim,5) for _ in range(1)]
        self.pool = nn.MaxPool1d(output_dim)
        self.mask_nn = [nn.Linear(in_features=step, out_features=step).to(device) for _ in range(3)]
        kernel_size = (5,2*self.enc_dim)
        self.encoder_mask2 = nn.Sequential(nn.Conv2d(in_channels = self.rapd_dim, out_channels =self.rapd_dim, kernel_size =5),
                            nn.ReLU())
        # self.mask_nn = [nn.Linear(in_features=step, out_features=step).to(device) for _ in range(3)]
    def _mask_init(self,input_dim, hidden_dim):
        W = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=hidden_dim, out_features=1)).to(device)
        return W

    def _masks_cnn_init(self, input_dim, output_dim, kernel_size):
        W = nn.Sequential(nn.Conv1d(in_channels = input_dim, out_channels =output_dim, kernel_size =kernel_size,padding =int((kernel_size-1)/2)),
                            nn.ReLU()).to(device)
        return W
    def _masks_cnn(self, X):
        X = X.permute(0,2,1)
        mask_matrixs = [_mask(X) for _mask in self.encoder_mask]
        mask_matrixs = [self.pool(mask_matrix.permute(0,2,1).to(device)) for mask_matrix in mask_matrixs]
        mask_matrixs = [_nn(mask_matrix.squeeze()) for _nn,mask_matrix in zip(self.mask_nn,mask_matrixs)]
        mask_matrixs = [mask_matrix.unsqueeze(-1) for mask_matrix in mask_matrixs]
        return mask_matrixs

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = Variable(torch.randn_like(z_mean)).to(device)

        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs,kl_loss      
    def init_edge2emb(self, nodetime2emb_maps):
        self.nodetime2emb_maps = nodetime2emb_maps
    def collect_rap_mapping(self, src_idx_l, cut_time_l, subgraph):
        src_neighbor, partners, partners_neighbor = subgraphs
        
        neighbor_n,neighbor_eid, neighbor_ts = src_neighbor
        partner_n, partner_eid, partner_ts= partners
        neighbor_pt_n, neighbor_pt_eid,neighbor_pt_ts = partners_neighbor
        batch_size = len(src_idx_l)

        for row in range(len(src_idx_l)):
            tag_key = str(e_idx_l[row])    
            tag_all = subgraph_all_node[row].astype(int)   
            
            target_neighors = src_2layer_node[row] # num_neighbor * num_partners
            src_3layer_node = subgraph_3layer_node[row] # num_neighbor * num_partners * num_neighbor 

            start = time.time()
            corr_value,doc_list_time,tfidf_time = self.tfidf2vec(tag_all_id, src_3layer_node)
            end= time.time()
            source.append(end-start)
            start = time.time()
            corr_tag_value,node_matrix_time,prob2vec_time = self.prob2vec(tag_all_id, src_3layer_node)
            end= time.time()
            dest.append(end-start)
            nodetime2emb[tag_key] = corr_value
            tag_key_node = tag_key+'-nodes'
            nodetime2emb[tag_key_node] = corr_tag_value
#         print('source', sum(source))
#         print('dest', sum(dest))
        return nodetime2emb

    def init_internal_data(self, src_idx_l, cut_time_l,subgraph):
        if self.enc_dim == 0:
            return
        self.nodetime2emb_maps = self.collect_rap_mapping(src_idx_l, cut_time_l, subgraph)

   
    def cut_to_num(self, ngh_idx,num_neighbor=32):
        if len(ngh_idx)>num_neighbor:
            ngh_idx_path = ngh_idx[-num_neighbor:]
        else:                
            insertzero = [0]*(num_neighbor-len(ngh_idx))
            ngh_idx_path = np.append(insertzero,ngh_idx)  
        return ngh_idx_path

    def reparameter(self, log_alpha, beta=1e-3, test=False):
        """ binary version of the Gumbel-Max trick
        """
        # print('log_alpha ',log_alpha)

        if not test:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha/beta
            gate_inputs = gate_inputs.sigmoid()
        return gate_inputs

    def generate_mask(self, x, _type='source'):
        if _type=='source':
            masks = [self.trainable_masks[i](x) for i in range(self.mask_num_src)]
        else:
            masks = [self.trainable_masks_dest[i](x) for i in range(self.mask_num_dst)]
        return masks

    def c_node_sim(self, seq_u, seq_partner, _time):
        node = seq_u[-1]
        t0, t1, t2, t3 = 0,0,0,0
        # start = time.time()
        # nb_v_list = list(set(np.array(seq_partner).flatten())|set(seq_u))
        nb_v_list = set(seq_u)
        for seq_p in seq_partner:
            nb_v_list.update(set(seq_p))

        v_score = dict()
        # print('nb_v_list ', nb_v_list)
        for ele in nb_v_list:
            #union_size, intersecton_size = self.graph.compute_union_and_intersection(node, ele, _time)
            union_size, intersection_size = self.graph.compute_small_union_and_intersection(node, ele, _time, self.rapd_dim)
            v_score[ele] = 1.0 * intersection_size / (union_size + 1e-8)
        seq_u_vector = [v_score[ele] for ele in seq_u]
        
        seq_partner_vector = []
        for seq in seq_partner:
            seq_partner_vector.append([v_score[ele] for ele in seq])
        return seq_u_vector, seq_partner_vector,(0,0,0,0)
    def node_sim(self, seq_u, seq_partner, tgt_neighbors, _eid): 
        nb_v = tgt_neighbors[str(seq_u[-1])+'_'+str(_eid)]
        nb_v_list = list(set(np.array(seq_partner).flatten())|set(seq_u))

        v_score = dict()
        start = time.time()
        for ele in nb_v_list:
            nb_vi = tgt_neighbors[str(ele)+'_'+str(_eid)]
            v_score[ele] = jaccard_distance(nb_v, nb_vi)
        # end = time.time()
        # print('jaccard_distance',end-start)
        # start = time.time()   
        seq_u_vector = [v_score[ele] for ele in seq_u]
        seq_partner_vector = []
        for seq in seq_partner:
            seq_partner_vector.append([v_score[ele] for ele in seq])
        return seq_u_vector, seq_partner_vector
    def source_vector_test(self, neighbor_n, tag_3layer_n, ts_l, e_idx_l, test):
        seq_len = len(tag_3layer_n[0][0][0])
        batch_user = []
        batch_partner = []
        mask_input = []
        for i, (nb, nb_part) in enumerate(zip(neighbor_n, tag_3layer_n)): 
            _eid = e_idx_l[i]
            _time = ts_l[i]
            user_seq = []
            partner_seq = []
            mask_seq = []
            start = time.time() 
            for j in range(self.step):

                sid = j+1
                eid = j+seq_len+1
                seq_u = nb[sid:eid]
                seq_partner = nb_part[j]
                seq_u_vector =[]
                for nb_vi in seq_partner:                
                    seq_u_vector.append(jaccard_distance(seq_u, nb_vi))
                user_seq.append(seq_u_vector)
            batch_user.append(user_seq)
        batch_user = torch.FloatTensor(np.array(batch_user)/self.enc_dim).to(device) 
        return batch_user, [], []

    def source_vector(self, neighbor_n, tag_3layer_n, ts_l, e_idx_l, test):
        seq_len = len(tag_3layer_n[0][0][0])
        batch_user = []
        batch_partner = []
        mask_input = []

        for i, (nb, nb_part) in enumerate(zip(neighbor_n, tag_3layer_n)): 
            _eid = e_idx_l[i]
            _time = ts_l[i]
            user_seq = []
            partner_seq = []
            mask_seq = []
            start = time.time() 

            for j in range(self.step):

                sid = j+1
                eid = j+seq_len+1
                seq_u = nb[sid:eid]
                seq_partner = nb_part[j]
                seq_u_vector, seq_partner_vector,t = self.c_node_sim(seq_u, seq_partner, _time)
                user_seq.append(seq_u_vector)
                partner_seq.append(seq_partner_vector)
            
            mean_seq = np.mean(partner_seq, axis=1)
            
            for seq_u_vector, seq_partner_vector, seq_partner_mean in zip(user_seq, partner_seq,mean_seq):
                mask_seq.append([seq_u_vector+p_vec+list(seq_partner_mean) for p_vec in seq_partner_vector])
            batch_user.append(user_seq)
            batch_partner.append(partner_seq)
            mask_input.append(mask_seq)
        batch_user = torch.FloatTensor(np.array(batch_user)).to(device) #batch_size*step*num_neighbor[0]
        batch_partner = torch.FloatTensor(np.array(batch_partner)).to(device)#batch_size*step*partner_size*num_neighbor[0]
        mask_input = torch.FloatTensor(np.array(mask_input)).to(device)

        mask_matrixs = self.generate_mask(mask_input.to(torch.float32))
        # print(mask_matrixs.shape)

        mask_reparas = [self.reparameter(mask_matrix, test=test) for mask_matrix in mask_matrixs]
        return batch_user, batch_partner, mask_reparas
    def c_node_sim_v(self, seq_u, src, _time):
        base_v = seq_u[-1]
        cmp_v = set(seq_u[:-1])
        user_sim = dict()
        v_score = dict()
        for v in cmp_v:
            #co_neighbor = self.graph.get_co_neighbors(base_v, v, _time)
            co_neighbor = self.graph.get_small_co_neighbors(base_v, v, _time, self.rapd_dim)
            for u in co_neighbor:
                if u not in user_sim:
                    #dist = self.c_node_sim(u, src, _time)
                    union_size, intersection_size = self.graph.compute_small_union_and_intersection(u, src, _time, self.enc_dim)
                    dist = 1.0 * intersection_size / (union_size + 1e-8)
                    user_sim[u] = dist
            prob_dist = [user_sim[u] for u in co_neighbor]
            histrange = np.arange(0,1.001,0.1)
            v_score[v], _ = np.histogram(prob_dist, bins=[0.0, 0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])

        des_vectors = []
        for v in seq_u[:-1]:
            des_vectors.append(v_score[v])
            
        return des_vectors
            
    def node_sim_v(self,seq_u, seq_partner,src_neighbors,tgt_neighbors, _eid): 
        nbs = [tgt_neighbors[str(ele)+'_'+str(_eid)] for ele in seq_u]
        nb_v = set(nbs[-1])
        seq_list = [set(ele)&nb_v for ele in nbs[:-1]]


        des_vectors = []
        for nb_set in seq_list:
            v_score = []
            # hist_vector = np.zeros(5)
            for ui in nb_set:     
                nb_ui = src_neighbors[str(ui)+'_'+str(_eid)]
                v_score.append(jaccard_distance(seq_u, nb_ui)) 

            hist_vector,_ = np.histogram(v_score, bins=[0.0, 0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
            des_vectors.append(hist_vector)

        return des_vectors

    def desti_vector(self, neighbor_n, tag_3layer_n, src_l, ts_l, e_idx_l, test):
        seq_len = len(tag_3layer_n[0][0][0])
        batch_user = []
        batch_partner = []
        mask_input = []
        for i, (nb, nb_part) in enumerate(zip(neighbor_n, tag_3layer_n)): 
            _eid = e_idx_l[i]
            src = src_l[i]
            _time = ts_l[i]

            user_seq = []
            partner_seq = []
            mask_seq = []
            start = time.time() 
            for j in range(0,self.step):
                sid = j+1
                eid = j+seq_len+1
                seq_u = nb[sid:eid]
                seq_partner = nb_part[j] 

                hist_vector = self.c_node_sim_v(seq_u, src, _time)
                partner_seq.append(hist_vector)

            batch_partner.append(partner_seq)
        batch_partner = np.array(batch_partner)
        batch_partner = torch.FloatTensor(batch_partner/self.rapd_dim).to(device)#batch_size*step*partner_size*num_neighbor[0]
        batch_mean = torch.mean(batch_partner,dim=-2, keepdim=True)
        rpsize = batch_partner.shape[-2]
        batch_mean =batch_mean.repeat((1,1,rpsize,1))
        batch_com = torch.cat([batch_partner, batch_mean],dim=-1)
        mask_matrixs = self.generate_mask(batch_com, _type='dest')


        mask_reparas = [self.reparameter(mask_matrix, test=test) for mask_matrix in mask_matrixs]
        return batch_partner, mask_reparas
    def forward(self, src_idx_l,dst_idx_l,e_idx_l, cut_time_l,subgraph, test):
        neighbor, partner, partner_neighbor = subgraph
        neighbor_n,neighbor_eid, neighbor_ts= neighbor
        partner_n, partner_eid, partner_ts = partner
        tag_3layer_n, tag_3layer_eid, tag_3layer_ts= partner_neighbor

        batch_nodes = np.array(neighbor_eid)
        edge_features = None
        if self.e_feat_dim !=0:
            L = len(batch_nodes[0])
            batch_nodes = batch_nodes[:,L-self.step:]
            batch_size,num_neighbor = len(batch_nodes), len(batch_nodes[0])
            batch_nodes = batch_nodes.flatten()
            unordered_encodings = np.array([self.nodetime2emb_maps[str(key)] for key in batch_nodes])
            encodings = unordered_encodings.reshape(batch_size,num_neighbor,-1)
            encodings = torch.FloatTensor(encodings).to(device)
            if not test:
                random_noise = (torch.rand(encodings.shape).to(device)-0.5)*0.4
                encodings = encodings+random_noise
            edge_features = self.trainable_embedding_feat(encodings)
        # representation of source node 
        start = time.time()
        if self.RAPs != 'None':
            s_vector, s_partner, s_mask = self.source_vector(neighbor_n,tag_3layer_n, cut_time_l, e_idx_l,test)
        else:
            s_vector, s_partner, s_mask = None, None, None
        if self.RAPd != 'None':
            d_partner, d_mask = self.desti_vector(neighbor_n, tag_3layer_n, src_idx_l, cut_time_l, e_idx_l, test)

        else:
            d_partner, d_mask = None,None
        return edge_features, (s_vector, s_partner, s_mask), (d_partner, d_mask)
        
