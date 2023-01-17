import logging
import time
import numpy as np
import torch
from torch.autograd import Variable
import multiprocessing as mp
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
from position import *
from numba import jit
import editdistance
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pylab import *
from collections import Counter
from scipy import spatial
from scipy.linalg import logm
from util_mod_c import  RAPEncoder

PRECISION = 5
POS_DIM_ALTER = 100
import os
os.environ['PYTHONHASHSEED'] = str(seed)


        
class SetPooler(nn.Module):
    """
    Implement similar ideas to the Deep Set
    """
    def __init__(self, n_features, out_features, dropout_p=0.1, walk_linear_out=False):
        super(SetPooler, self).__init__()
        self.mean_proj = nn.Linear(n_features, n_features)
        self.max_proj = nn.Linear(n_features, n_features)
        self.attn_weight_mat = nn.Parameter(torch.zeros((2, n_features, n_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.attn_weight_mat.data[0])
        nn.init.xavier_uniform_(self.attn_weight_mat.data[1])
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Sequential(nn.Linear(n_features, out_features), nn.ReLU(), self.dropout)
        self.walk_linear_out = walk_linear_out

    def forward(self, X, agg='sum'):
        if agg=='max':  
            X, ind = torch.max(X,dim=-2)
            return self.out_proj(X)
        if agg == 'sum':
            return self.out_proj(X.sum(dim=-2))
        if agg == 'None':
            return self.out_proj(X)
        else:
            assert(agg == 'mean')
            return self.out_proj(X.mean(dim=-2))


class RAPAD(torch.nn.Module):
    def __init__(self, n_feat, e_feat,tf_matrix, drop_out=0.1, num_neighbors=20, cpu_cores=1, verbosity=1,   
        get_checkpoint_path=None, v_size=32883, time_dim=5, dist_dim=10, ngram = 64, raps='on', rapd='on', step=5,mask_num=7):
        super(RAPAD, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.verbosity = verbosity

        # subgraph extraction hyper-parameters
        self.num_neighbors, self.num_layers = process_sampling_numbers(num_neighbors, 2)
        self.step =step
        self.ngh_finder = None
        self.tf_matrix = tf_matrix
        self.src_embed = None
        self.inter_result = dict()
        self.e_feat_dim = None
        self.cent_emb = None
        self.raps = raps
        self.rapd = rapd

        svec_dim= self.num_neighbors[0]
        dvec_dim = dist_dim
        self.dist_dim = dist_dim
        # features
        if n_feat is not None:
            self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
            self.n_feat_dim = self.n_feat_th.shape[1]  # node feature dimension
            self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        else:
            self.n_feat_th = None
            self.n_feat_dim = 0
        if e_feat is not None:
            self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)
            self.e_feat_dim =self.e_feat_th.shape[1]   # edge feature dimension
            self.e_feat_o = 10
            self.edge_raw_embed = nn.Embedding(num_embeddings=len(self.e_feat_th), embedding_dim=self.e_feat_dim)
            self.feat_encoder = nn.Sequential(nn.Linear(in_features=self.e_feat_dim*2, out_features=self.e_feat_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.e_feat_dim, out_features=self.e_feat_o))

            # self.e_feat_dim = 10
            gru_dim_s = 2*svec_dim#+self.e_feat_dim
            self.out_edge = nn.Sequential(nn.Linear(in_features=svec_dim*3+self.e_feat_dim, out_features=svec_dim),
                                                     nn.ReLU(), nn.Dropout(0.1))


            gru_dim_d = dvec_dim#+self.e_feat_dim
            input_dim = svec_dim*2+gru_dim_d+10#self.e_feat_dim 
            input_dim_s = gru_dim_s+self.e_feat_dim  
            input_dim_d = gru_dim_d+self.e_feat_dim  
             
        else:
            self.e_feat_th = None
            self.e_feat_dim = 0


            self.e_feat_o = self.e_feat_dim
            gru_dim_s = svec_dim*2
             
            gru_dim_d = dvec_dim
            input_dim = svec_dim*2+gru_dim_d

            # gru_dim_s = 16*2

            self.out_edge = nn.Sequential(nn.Linear(in_features=svec_dim*3, out_features=svec_dim),
                                                     nn.ReLU(), nn.Dropout(0.1))

        self.time_dim = time_dim  
        self.edge_sd = nn.Sequential(nn.Linear(in_features=gru_dim_s, out_features=gru_dim_s),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_s, out_features=gru_dim_s)) 
        self.edge_s_out = nn.Sequential(nn.Linear(in_features=gru_dim_s, out_features=gru_dim_s),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_s, out_features=gru_dim_s)) 
        self.edge_sd2 = nn.Sequential(nn.Linear(in_features=gru_dim_d, out_features=gru_dim_d),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_d, out_features=gru_dim_d))
        self.edge_com = nn.Sequential(nn.Linear(in_features=gru_dim_s+gru_dim_d+self.e_feat_o, out_features=gru_dim_s+gru_dim_d),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_s+gru_dim_d, out_features=gru_dim_s+gru_dim_d))
        kernel_size = 3
        feature_size = 100
        self.mask_cnn = nn.Sequential(nn.Conv1d(in_channels = svec_dim, out_channels =feature_size, kernel_size =kernel_size),
                            nn.ReLU(),
                            nn.MaxPool1d(self.num_neighbors[1]-kernel_size+1))
        self.mask_fc = nn.Sequential(nn.Linear(in_features=feature_size, out_features=svec_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=svec_dim, out_features=svec_dim))

        kernel_size2 = 3
        self.mask_cnn_dst = nn.Sequential(nn.Conv1d(in_channels = dvec_dim, out_channels =feature_size, kernel_size =kernel_size2),
                            nn.ReLU(),
                            nn.MaxPool1d(self.num_neighbors[0]-kernel_size2))
        self.mask_fc_dst = nn.Sequential(nn.Linear(in_features=feature_size, out_features=dvec_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=dvec_dim, out_features=dvec_dim))        
        
        self.dropout = nn.Dropout(0.1)
        print('gru_dim_s', gru_dim_s)

        if raps == 'None':
            self.rap_dim = self.num_neighbors[0]
            self.model_dim = self.e_feat_o + self.n_feat_dim + self.time_dim + self.num_neighbors[0]
            self.encoder_model = nn.GRU(input_size=gru_dim_d, hidden_size=(gru_dim_d)//2, batch_first=True, bidirectional=True)
            self.out_sequential = nn.Sequential(nn.Linear(in_features=gru_dim_d, out_features=gru_dim_d),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_d, out_features=1),
                                                     nn.Sigmoid())
            self.pooler = SetPooler(n_features=gru_dim_d, out_features=gru_dim_d, dropout_p=0.1, walk_linear_out=False)
            self.projector = nn.Sequential(nn.Linear(gru_dim_d, gru_dim_d), 
                                       nn.ReLU())

        elif rapd == 'None':
            self.rap_dim = self.num_neighbors[1]
            self.model_dim = self.e_feat_o + self.n_feat_dim + self.time_dim + self.num_neighbors[1]
            self.encoder_model = nn.GRU(input_size=gru_dim_s, hidden_size=(gru_dim_s)//2, batch_first=True, bidirectional=True)
            self.pooler = SetPooler(n_features=gru_dim_s, out_features=gru_dim_s, dropout_p=0.1, walk_linear_out=False)
            self.projector = nn.Sequential(nn.Linear(gru_dim_s, gru_dim_s), 
                                       nn.ReLU())
            self.out_sequential = nn.Sequential(nn.Linear(in_features=gru_dim_s, out_features=gru_dim_s),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_s, out_features=1),
                                                     nn.Sigmoid())
        else:
            self.rap_dim = self.num_neighbors[0] + self.num_neighbors[1]
            self.model_dim = self.e_feat_o + self.n_feat_dim + self.time_dim + self.num_neighbors[0] + self.num_neighbors[1]
            self.sd_combine = nn.Sequential(nn.Linear(in_features=svec_dim*3, out_features=svec_dim),
                                                     nn.ReLU(), nn.Dropout(0.1))
            self.out_sequential = nn.Sequential(nn.Linear(in_features=gru_dim_s+gru_dim_d, out_features=gru_dim_s+gru_dim_d),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_s+gru_dim_d, out_features=1),
                                                     nn.Sigmoid())
            self.encoder_model = nn.GRU(input_size=gru_dim_s+gru_dim_d, hidden_size=(gru_dim_s+gru_dim_d)//2, batch_first=True, bidirectional=True)
            # self.encoder_model = nn.LSTM(input_size=gru_dim_s+gru_dim_d, hidden_size=(gru_dim_s+gru_dim_d)//2, batch_first=True, bidirectional=True)
            # encoder_layer = nn.TransformerEncoderLayer(d_model=gru_dim_s+gru_dim_d, nhead=2)
            # self.encoder_model = nn.TransformerEncoder(encoder_layer, num_layers=3)

            self.pooler = SetPooler(n_features=gru_dim_s+gru_dim_d, out_features=gru_dim_s+gru_dim_d, dropout_p=0.1, walk_linear_out=False)
            self.projector = nn.Sequential(nn.Linear(gru_dim_s+gru_dim_d, gru_dim_s+gru_dim_d), 
                                       nn.ReLU())
            self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        # self.model_dim = self.n_feat_dim + self.e_feat_dim + self.time_dim + self.pos_dim + self.num_neighbors[1]
        self.logger.info('training ration 0.05')
        self.logger.info('step: {}, neighbors: {}, node dim: {}, edge dim: {}, time dim: {}'.format(self.step, self.num_neighbors, self.n_feat_dim, self.e_feat_dim, self.time_dim))

        # dropout for both tree and walk based model
        self.dropout_p = drop_out
        self.v_size = v_size
        edge_dim = self.num_neighbors[0] + self.num_neighbors[1]

        self.mask_num = mask_num
        
        self.corr_encoder = RAPEncoder(tf_matrix=self.tf_matrix, v_size=v_size, enc_dim=self.num_neighbors[0], num_layers=self.num_layers, ngh_finder=self.ngh_finder,
                                                cpu_cores=cpu_cores, verbosity=verbosity, logger=self.logger, rapd_dim=self.num_neighbors[1], 
                                                e_feat_dim=self.e_feat_dim,RAPs=raps, RAPd=rapd,dist_dim=dist_dim,step=step,mask_num=mask_num)
        
     
        
        self.edge_s = nn.Sequential(nn.Linear(in_features=svec_dim, out_features=svec_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=svec_dim, out_features=svec_dim))
        self.edge_d = nn.Sequential(nn.Linear(in_features=dvec_dim, out_features=dvec_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=dvec_dim, out_features=dvec_dim))

        
        self.edge_sd_out = nn.Sequential(nn.Linear(in_features=edge_dim*4, out_features=edge_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=edge_dim, out_features=edge_dim))
        self.edge_cent = nn.Sequential(nn.Linear(in_features=svec_dim, out_features=svec_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=svec_dim, out_features=svec_dim))
        
        

        self.get_checkpoint_path = get_checkpoint_path
        if self.mask_num == 0:
            self.mask_model = False
        else:
            self.mask_model = True

        print('mask encoder ', self.mask_model)
        print('mask_num ', self.mask_num)


    def concrete_sample(self, log_alpha, beta=3.0, test=False):
        """ binary version of the Gumbel-Max trick
        """
        if not test:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()
        return gate_inputs
    def init_center(self,src_idx_l,dst_l_cut,e_idx_l, cut_time_l):
        subgraphs = self.ngh_finder.get_subgraphs(src_idx_l,dst_l_cut, cut_time_l, num_neighbors=self.num_neighbors,step=self.step, e_idx_l=e_idx_l)
        rap_features,_,_ = self.retrieve_corr_features(src_idx_l,dst_l_cut,e_idx_l, cut_time_l,subgraphs, test=True)               
        _,edge_center = self.edge_update(rap_features)
        return edge_center
    def init_enc_model(self):
        enc_model = EncoderModel(rap_dim=self.rap_dim, model_dim=self.model_dim, out_dim=self.model_dim,
                                                     dropout_p=self.dropout_p, logger=self.logger, step=self.step, encode_type=self.encodetype)
        return enc_model

    
    def contrast(self, src_idx_l,dst_l_cut, cut_time_l, e_idx_l=None, test=False):
        self.test = test
        self.ngh_finder.zeros_partners = 0
        self.ngh_finder.zeros_partners_or = 0
        subgraph_src = self.ngh_finder.get_interaction(src_idx_l,dst_l_cut, cut_time_l, num_neighbors=self.num_neighbors,step=self.step, e_idx_l=e_idx_l)
        score,kl_loss = self.forward(src_idx_l,dst_l_cut,e_idx_l, cut_time_l, subgraph_src, test=test)
        self.flag_for_cur_edge = False

        return score,kl_loss

    def forward(self, src_idx_l,dst_l_cut,e_idx_l, cut_time_l, subgraphs=None, test=False):
        src_embed,kl_loss = self.forward_msg(src_idx_l,dst_l_cut,e_idx_l, cut_time_l, subgraphs, test=test)
        return src_embed,kl_loss

    def forward_msg(self, src_idx_l,dst_l_cut,e_idx_l, cut_time_l, subgraphs, test=False):
        encoder = self.corr_encoder
        edge_features, source, desti = encoder(src_idx_l,dst_l_cut,e_idx_l, cut_time_l,subgraphs, test=test)           
        if self.raps == 'None':
            score  = self.edge_update_d(desti[0], edge_features, desti[1])
        elif self.rapd == 'None':
            score  = self.edge_update_s(source[0], source[1],edge_features, source[2])
        else:
            score = self.edge_update(source, desti, edge_features)     
        return score,0
    def edge_update(self, source, desti, encodings):
        s_vector, s_partner, s_mask = source
        d_partner, d_mask = desti
        scores = []
        s_features, d_features = [], []

        if self.num_neighbors[0]<3:
            d_mask = []
        elif self.num_neighbors[1]<3:
            s_mask = []
        if len(s_mask)>0:
            L = s_partner.shape[-2]
            for mask in s_mask:
                hk_user = self.edge_s(s_vector) 
                X = s_partner*mask
                hk_center = []
                for partner in X:
                    # print(partner.shape)
                    partner = partner.permute(0,2,1)
                    partner = self.mask_cnn(partner)
                    # print(partner.shape)
                    partner = partner.squeeze(dim=-1)
                    partner = self.mask_fc(partner)
                    hk_center.append(partner)
                hk_center = torch.stack(hk_center)
                edge_feature = torch.cat([hk_user, hk_center], dim=-1)
                edge_feature_s = self.edge_sd(edge_feature)

                s_features.append(edge_feature_s)
        else:
            X = s_partner
            hk_center = X.mean(-2)
            hk_center = self.edge_cent(hk_center)
            hk_user = self.edge_s(s_vector) 
            edge_feature = torch.cat([hk_user, hk_center], dim=-1)
            edge_feature_s = self.edge_sd(edge_feature)
            s_features.append(edge_feature_s)

        if len(d_mask) > 0:
            for mask in d_mask:             
                # AERd
                X = d_partner
                hk_center = []
                X = X*mask
                div = mask.sum(dim=-2)
                for partner in X:
                    # print(partner.shape)
                    partner2 = partner.permute(0,2,1)                  
                    partner_m = self.mask_cnn_dst(partner2)
                    partner1 = partner_m.squeeze(dim=-1)
                    partner = self.mask_fc_dst(partner1)
        
                    hk_center.append(partner)
                hk_center = torch.stack(hk_center)
                edge_feature_d = torch.cat([hk_center], dim=-1)
                d_features.append(edge_feature_d)       
        else:
            X = d_partner
            hk_center = X.mean(-2)
            hk_center = self.edge_d(hk_center)
            edge_feature_d = torch.cat([hk_center], dim=-1)
            d_features.append(edge_feature_d)

        for edge_feature_s, edge_feature_d in zip(s_features, d_features):
            if self.e_feat_dim !=0:
                edge_feature = torch.cat([edge_feature_s, edge_feature_d,encodings], dim=-1)
            else:
                edge_feature = torch.cat([edge_feature_s, edge_feature_d], dim=-1)
            edge_feature = self.edge_com(edge_feature)
            
            X = edge_feature.permute([1, 0, 2]) #X [S, B, N]
            encoded_feature = self.encoder_model(X)[0]
            encoded_feature = self.dropout(encoded_feature.permute([1, 0, 2]))
            encoded_feature = self.projector(encoded_feature)          
            encoded_feature = self.pooler(encoded_feature, agg='mean')
            score = self.out_sequential(encoded_feature) 
            scores.append(score)
        scores = torch.cat(scores, dim=-1)
        score,_ = torch.min(scores, dim=-1,keepdim=True)
        score = score.squeeze()    
        return score

    def edge_update_d(self, batch_partner, encodings, masks=None):
        d_partner  = batch_partner
        d_mask = masks
        d_features = []
        if len(d_mask) > 0:
            for mask in d_mask:             
                # AERd
                X = d_partner
                hk_center = []
                for partner in X:
                    partner = partner.permute(0,2,1)
                    partner = self.mask_cnn_dst(partner)
                    partner = partner.squeeze()
                    partner = self.mask_fc_dst(partner)
                    hk_center.append(partner)
                hk_center = torch.stack(hk_center)
                edge_feature_d = torch.cat([hk_center], dim=-1)
                d_features.append(edge_feature_d)       
        else:
            X = d_partner
            hk_center = X.mean(-2)
            hk_center = self.edge_d(hk_center)
            edge_feature_d = torch.cat([hk_center], dim=-1)
            d_features.append(edge_feature_d)
        scores = []
        for edge_feature_d in d_features:
            if self.e_feat_dim !=0:
                edge_feature = torch.cat([edge_feature_d,encodings], dim=-1)
            edge_feature = self.edge_sd2(edge_feature)
            X = edge_feature.permute([1, 0, 2]) #X [S, B, N]
            # print(X.shape)
            encoded_feature = self.encoder_model(X)[0]
            # print(encoded_feature.shape)
            encoded_feature = self.dropout(encoded_feature.permute([1, 0, 2]))
            encoded_feature = self.projector(encoded_feature)          
            encoded_feature = self.pooler(encoded_feature, agg='mean')

            # hidden_feature,encoded_feature = self.encoder_model(X)
            # print(encoded_feature.shape)
            score = self.out_sequential(encoded_feature) 
            scores.append(score)
        scores = torch.cat(scores, dim=-1)
        score,_ = torch.min(scores, dim=-1,keepdim=True)
        score = score.squeeze()            
        return score
    def edge_update_s(self, batch_user, batch_partner, encodings, masks=None):
        b_size, step, p_size,seq_size = batch_partner.shape
        # print(masks.squeeze())
        # print(batch_partner)
        # masks = None
        s_vector, s_partner  = batch_user, batch_partner
        s_features = []
        s_mask = masks
        if len(s_mask)>0:
            L = s_partner.shape[-2]
            for mask in s_mask:
                hk_user = self.edge_s(s_vector) 
                X = s_partner*mask
                hk_center = []
                for partner in X:
                    partner = partner.permute(0,2,1)
                    partner = self.mask_cnn(partner)
                    partner = partner.squeeze()
                    partner = self.mask_fc(partner)
                    hk_center.append(partner)
                hk_center = torch.stack(hk_center)
                if self.e_feat_dim !=0:
                    edge_feature = torch.cat([hk_user, hk_center, encodings], dim=-1)
                else:
                    edge_feature = torch.cat([hk_user, hk_center], dim=-1)
                    # print(edge_feature.shape)
                edge_feature_s = self.edge_sd(edge_feature)

                s_features.append(edge_feature_s)
        else:
            X = s_partner
            hk_center = X.mean(-2)
            hk_center = self.edge_cent(hk_center)
            hk_user = self.edge_s(s_vector) 
            if self.e_feat_dim !=0:
                edge_feature = torch.cat([hk_user, hk_center, encodings], dim=-1)
            else:
                edge_feature = torch.cat([hk_user, hk_center], dim=-1)
            edge_feature_s = self.edge_sd(edge_feature)
            s_features.append(edge_feature_s)
        scores = []
        for edge_feature  in s_features:
            X = edge_feature.permute([1, 0, 2]) #X [S, B, N]
            encoded_feature = self.encoder_model(X)[0]
            encoded_feature = self.dropout(encoded_feature.permute([1, 0, 2]))
            encoded_feature = self.projector(encoded_feature)          
            encoded_feature = self.pooler(encoded_feature, agg='mean')

            score = self.out_sequential(encoded_feature) 
            scores.append(score)
        scores = torch.cat(scores, dim=-1)
        score,_ = torch.min(scores, dim=-1,keepdim=True)
        score = score.squeeze()         
        return score


    def init_hidden_embeddings(self, node_records):
        if self.n_feat_th is not None:
            device = self.n_feat_th.device
            node_records_th = torch.from_numpy(node_records).long().to(device)
            hidden_embeddings = self.node_raw_embed(node_records_th) 
        else:
            hidden_embeddings = None
        return hidden_embeddings


    def retrieve_corr_features(self, src_idx_l,dst_idx_l,e_idx_l, cut_time_l,subgraph, test=False):
        encode = self.corr_encoder
        position_features, mask_repara, kl_loss = encode(src_idx_l,dst_idx_l,e_idx_l, cut_time_l,subgraph, test=test)
        return position_features,mask_repara,kl_loss

    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder
        self.corr_encoder.ngh_finder = ngh_finder


