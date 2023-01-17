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
from util_mod_c import MultiHeadAttention, GRUModel, PosEncode, EmptyEncode, TimeEncode, RAPEncoder

PRECISION = 5
POS_DIM_ALTER = 100
import os
os.environ['PYTHONHASHSEED'] = str(seed)


class EncoderModel(nn.Module):
    def __init__(self, rap_dim, model_dim, out_dim, logger,  dropout_p=0.1, step=10, encode_type='Att'):
        '''
        masked flags whether or not use only valid temporal walks instead of full walks including null nodes
        '''
        super(EncoderModel, self).__init__()
        self.rap_dim = rap_dim
        self.model_dim = model_dim
        self.attn_dim = self.model_dim//2  # half the model dim to save computation cost for attention
        self.out_dim = out_dim
        self.dropout_p = dropout_p
        self.logger = logger
        self.encode_type = encode_type

        self.feature_encoder = FeatureEncoder(self.model_dim, self.model_dim, self.dropout_p, step=step, encode_type = encode_type)  # encode all types of features along each temporal walk
        # print('self.feature_encoder.model_dim+rap_dim', self.feature_encoder.model_dim+self.rap_dim)
        if self.encode_type == 'Att':
            self.projector = nn.Sequential(nn.Linear(self.feature_encoder.model_dim+self.rap_dim, self.attn_dim),  # notice that self.feature_encoder.model_dim may not be exactly self.model_dim is its not even number because of the usage of bi-lstm
                                       nn.ReLU(), nn.Dropout(self.dropout_p))  # TODO: whether to add #[, nn.Dropout())]?
        else:
            self.projector = nn.Sequential(nn.Linear(self.feature_encoder.model_dim, self.attn_dim),  # notice that self.feature_encoder.model_dim may not be exactly self.model_dim is its not even number because of the usage of bi-lstm
                                       nn.ReLU(), nn.Dropout(self.dropout_p))  # TODO: whether to add #[, nn.Dropout())]?

        self.pooler = SetPooler(n_features=self.attn_dim, out_features=self.out_dim, dropout_p=self.dropout_p, walk_linear_out=False)

    def forward_one_node(self, hidden_embeddings, time_features, edge_features, position_features,rap_features, masks=None):
        combined_features = self.aggregate(hidden_embeddings, time_features, edge_features, position_features, rap_features)
        combined_features = self.feature_encoder(combined_features, masks)    
        if self.encode_type == 'Att':
            rap0 = rap_features[:,-1,:].unsqueeze(1)
            combined_features = torch.cat([combined_features, rap0], dim=-1)
        # print(self.feature_encoder.model_dim+self.rap_dim)
        # print(combined_features.shape)
        X = self.projector(combined_features)
        X = self.pooler(X, agg='None')  # we are actually doing mean pooling since sum has numerical issues
        return X

    def aggregate(self, hidden_embeddings, time_features, edge_features, position_features,rap_features):
        device = rap_features.device
        if (edge_features is None) and (hidden_embeddings is None):
            combined_features = torch.cat([time_features, position_features, rap_features], dim=-1)
        elif hidden_embeddings is None:
            combined_features = torch.cat([time_features, position_features, rap_features, edge_features], dim=-1)
        elif edge_features is None:
            combined_features = torch.cat([time_features, position_features, rap_features, hidden_embeddings], dim=-1)
        else:
            combined_features = torch.cat([hidden_embeddings, time_features, edge_features, position_features,rap_features], dim=-1)
                    
        combined_features = combined_features.to(device)
        assert(combined_features.size(-1) == self.model_dim)
        return combined_features

class FeatureEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1,step=10, encode_type = 'Att'):
        super(FeatureEncoder, self).__init__()
        self.hidden_features_one_direction = hidden_features//2
        self.model_dim = self.hidden_features_one_direction * 2  # notice that we are using bi-lstm
        if self.model_dim == 0:  # meaning that this encoder will be use less
            return
        n_head = 2
        # print('in_features',in_features)
        self.encode_type = encode_type
        if encode_type == 'Att':
            self.encoder_model = MultiHeadAttention(n_head, d_model=self.hidden_features_one_direction, 
                                             d_k=in_features // n_head, 
                                             d_v=in_features // n_head, 
                                             dropout=dropout)
        elif encode_type == 'mask':
            self.encoder_model = GRUModel(input_num=in_features, hidden_num=in_features, output_num=in_features)
        elif encode_type == 'CNN':
            self.encoder_model = nn.Conv1d(in_channels =step, out_channels =hidden_features, kernel_size =3,padding =1)
        else:
            self.encoder_model = nn.LSTM(input_size=in_features, hidden_size=self.hidden_features_one_direction, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        batch, n_walk, feat_dim = X.shape
        # self.encode_type = 'Att'
        if self.encode_type == 'Att':
            q, k = X[:,-1,:], X[:,:-1,:]
            q = torch.unsqueeze(q, dim=1)      
            mask = mask[:,1:]
            mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
            mask = mask.permute([0, 2, 1]) #mask [B, 1, N]
            encoded_features,weight = self.encoder_model(q=q, k=k,v=k, mask=mask)
        elif self.encode_type == 'mask':
            X = X.permute([1, 0, 2]) #X [S, B, N]
            mask = mask.permute([1, 0, 2]) #mask [S, B, 1]
            hidden_features,encoded_features = self.encoder_model(X,mask=mask)
        elif self.encode_type == 'CNN':
            mask = None
            # print(X.shape)
            encoded_features = self.encoder_model(X)           
            encoded_features = encoded_features.permute(0,2,1)
            encoded_features = self.dropout(encoded_features)
        else:
            mask = None
            encoded_features = self.encoder_model(X)[0]
            encoded_features = self.dropout(encoded_features)
        return encoded_features

        
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
    def __init__(self, n_feat, e_feat,tf_matrix, drop_out=0.1, num_neighbors=20, cpu_cores=1, verbosity=1,   get_checkpoint_path=None, 
        encode_type='Att', v_size=32883, time_dim=5, dist_dim=10, ngram = 64, raps='tfidf', rapd='pair', step=5,mask_num=7):
        super(RAPAD, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.verbosity = verbosity
        self.encodetype = encode_type

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
            self.e_feat_o = self.e_feat_dim
            self.edge_raw_embed = nn.Embedding(num_embeddings=len(self.e_feat_th), embedding_dim=self.e_feat_dim)
            self.feat_encoder = nn.Sequential(nn.Linear(in_features=self.e_feat_dim*2, out_features=self.e_feat_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.e_feat_dim, out_features=self.e_feat_o))

            gru_dim_s = 2*svec_dim+self.e_feat_dim
            gru_dim_d = dvec_dim+self.e_feat_dim
                   
             
        else:
            self.e_feat_th = None
            self.e_feat_dim = 0
            self.e_feat_o = self.e_feat_dim
            

            gru_dim_s = svec_dim*2            
            gru_dim_d = dvec_dim

        self.time_dim = time_dim  # default to be time feature dimension
        
        
        self.edge_sd = nn.Sequential(nn.Linear(in_features=gru_dim_s, out_features=gru_dim_s),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_s, out_features=gru_dim_s)) 
        self.edge_s_out = nn.Sequential(nn.Linear(in_features=gru_dim_s, out_features=gru_dim_s),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_s, out_features=gru_dim_s)) 
        self.edge_sd2 = nn.Sequential(nn.Linear(in_features=gru_dim_d, out_features=gru_dim_d),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=gru_dim_d, out_features=gru_dim_d))
        self.edge_com = nn.Sequential(nn.Linear(in_features=gru_dim_s+gru_dim_d, out_features=gru_dim_s+gru_dim_d),
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
        # self.model_dim = self.n_feat_dim + self.num_neighbors[1] + self.time_dim + self.pos_dim + self.num_neighbors[1]
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
            self.encoder_model = nn.LSTM(input_size=gru_dim_s+gru_dim_d, hidden_size=(gru_dim_s+gru_dim_d)//2, batch_first=True, bidirectional=True)
            # self.encoder_model = nn.LSTM(input_size=gru_dim_s, hidden_size=(gru_dim_s)//2, batch_first=True, bidirectional=True)
            print('d_model',gru_dim_s+gru_dim_d)
            # encoder_layer = nn.TransformerEncoderLayer(d_model=gru_dim_s+gru_dim_d, nhead=2)
            # self.encoder_model = nn.TransformerEncoder(encoder_layer, num_layers=3)

            self.pooler = SetPooler(n_features=gru_dim_s+gru_dim_d, out_features=gru_dim_s+gru_dim_d, dropout_p=0.1, walk_linear_out=False)
            self.projector = nn.Sequential(nn.Linear(gru_dim_s+gru_dim_d, gru_dim_s+gru_dim_d), 
                                       nn.ReLU())
            self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        # self.model_dim = self.n_feat_dim + self.e_feat_dim + self.time_dim + self.pos_dim + self.num_neighbors[1]
        self.logger.info('training ration 0.05')
        self.logger.info('step: {}, neighbors: {}, node dim: {}, edge dim: {}, time dim: {}'.format(self.step, self.num_neighbors, self.n_feat_dim, self.e_feat_o, self.time_dim))


        # dropout for both tree and walk based model
        self.dropout_p = drop_out
        self.v_size = v_size
        edge_dim = self.num_neighbors[0] + self.num_neighbors[1]

        # encoders
        self.time_encoder = self.init_time_encoder('time', seq_len=step)
        self.pos_encoder = self.init_time_encoder('pos', seq_len=step)
        # print(self.e_feat_dim)
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
        self.enc_model = self.init_enc_model()
        
        

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

    def init_time_encoder(self, use_time, seq_len):
        if use_time == 'time':
            self.logger.info('Using time encoding')
            time_encoder = TimeEncode(expand_dim=self.time_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            time_encoder = PosEncode(expand_dim=self.time_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            time_encoder = EmptyEncode(expand_dim=self.time_dim)
        else:
            raise ValueError('invalid time option!')
        return time_encoder
    def contrast(self, src_idx_l,dst_l_cut, cut_time_l, e_idx_l=None, test=False):
        start = time.time()
        subgraph_src = self.ngh_finder.get_interaction(src_idx_l,dst_l_cut, cut_time_l, num_neighbors=self.num_neighbors,step=self.step, e_idx_l=e_idx_l)
        score,kl_loss = self.forward(src_idx_l,dst_l_cut,e_idx_l, cut_time_l, subgraph_src, test=test)
        self.flag_for_cur_edge = False

          
        return score,kl_loss 
    def forward(self, src_idx_l,dst_l_cut,e_idx_l, cut_time_l, subgraphs=None, test=False):
        start = time.time()
        src_embed,kl_loss = self.forward_msg(src_idx_l,dst_l_cut,e_idx_l, cut_time_l, subgraphs, test=test)
        # edge_features, source, desti = encoder(src_idx_l,dst_l_cut,e_idx_l, cut_time_l,subgraphs, test=test)

        end = time.time()
        return src_embed,kl_loss

    def forward_msg(self, src_idx_l,dst_l_cut,e_idx_l, cut_time_l, subgraphs, test=False):
        start = time.time()
        encoder = self.corr_encoder
        edge_features, source, desti = encoder(src_idx_l,dst_l_cut,e_idx_l, cut_time_l,subgraphs, test=test)           
        end = time.time()
        # print('retrieve_corr_features',end-start)
        start = time.time()
        if self.raps == 'None':
            score  = self.edge_update_d(desti[0], edge_features, desti[1])
        elif self.rapd == 'None':
            score  = self.edge_update_s(source[0], source[1],edge_features, source[2])
        else:
            score = self.edge_update(source, desti, edge_features)
        

        end = time.time()
        # print('edge_update',end-start)
        start = time.time()       
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
                # div = mask.sum(dim=-2)
                # hk_center = X.sum(-2)/(div+1e-8)
                # hk_center = self.edge_s(hk_center)
                hk_center = []
                for partner in X:
                    # print(partner.shape)
                    partner = partner.permute(0,2,1)
                    partner = self.mask_cnn(partner)
                    # print(partner.shape)
                    partner = partner.squeeze(dim=-1)
                    partner = self.mask_fc(partner)
                    hk_center.append(partner)
                    # print(partner.shape)
                    # exit()
                hk_center = torch.stack(hk_center)
                if self.e_feat_dim !=0:
                    edge_feature = torch.cat([hk_user, hk_center, encodings], dim=-1)
                else:
                    # print(hk_user.shape,hk_center.shape)
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

        if len(d_mask) > 0:
            for mask in d_mask:             
                # AERd
                X = d_partner
                hk_center = []
                X = X*mask
                div = mask.sum(dim=-2)
                # hk_center = X.sum(-2)/(div+1e-8)
                # hk_center = self.edge_d(hk_center)
                for partner in X:
                    # print(partner.shape)
                    partner2 = partner.permute(0,2,1)                  
                    partner_m = self.mask_cnn_dst(partner2)
                    partner1 = partner_m.squeeze(dim=-1)
                    # if self.num_neighbors[0] == 14:
                    # print(partner2.shape)
                    # print(partner_m.shape)
                    # print(partner1.shape)
                    # exit()
                    try:
                        partner = self.mask_fc_dst(partner1)
                    except:
                        print(partner2.shape)
                        print(partner_m.shape)
                        print(partner1.shape)
                        print(partner.shape)
                        exit()
                    hk_center.append(partner)
                hk_center = torch.stack(hk_center)
                if self.e_feat_dim !=0:
                    edge_feature = torch.cat([hk_center, encodings], dim=-1)
                    edge_feature_d = self.edge_sd2(edge_feature)
                else:
                    edge_feature_d = torch.cat([hk_center], dim=-1)
                d_features.append(edge_feature_d)       
        else:
            X = d_partner
            hk_center = X.mean(-2)
            hk_center = self.edge_d(hk_center)
            if self.e_feat_dim !=0:
                edge_feature = torch.cat([hk_center, encodings], dim=-1)
                edge_feature_d = self.edge_sd2(edge_feature)
            else:
                edge_feature_d = torch.cat([hk_center], dim=-1)
            d_features.append(edge_feature_d)

        for edge_feature_s, edge_feature_d in zip(s_features, d_features):
            edge_feature = torch.cat([edge_feature_s, edge_feature_d], dim=-1)
            edge_feature = self.edge_com(edge_feature)
            X = edge_feature.permute([1, 0, 2]) #X [S, B, N]
            # hidden_feature,encoded_feature = self.encoder_model(X)
            # score = self.out_sequential(encoded_feature) 
            # print(X.shape)
            # X = edge_feature
            encoded_feature = self.encoder_model(X)[0]
            # print(encoded_feature.shape)
            encoded_feature = self.dropout(encoded_feature.permute([1, 0, 2]))
            encoded_feature = self.projector(encoded_feature)          
            encoded_feature = self.pooler(encoded_feature, agg='mean')
            score = self.out_sequential(encoded_feature) 
            # print(score.shape)
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
                # div = mask.sum(dim=-2)
                # X = X*mask
                # hk_center = X.sum(-2)/(div+1e-8)
                # hk_center = self.edge_d(hk_center)
                for partner in X:
                    # print(partner.shape)
                    partner = partner.permute(0,2,1)
                    # print(partner.shape)
                    partner = self.mask_cnn_dst(partner)
                    # print(partner.shape)
                    partner = partner.squeeze()
                    partner = self.mask_fc_dst(partner)
                    hk_center.append(partner)
                hk_center = torch.stack(hk_center)
                if self.e_feat_dim !=0:
                    edge_feature = torch.cat([hk_center, encodings], dim=-1)
                    edge_feature_d = self.edge_sd2(edge_feature)
                else:
                    edge_feature_d = torch.cat([hk_center], dim=-1)
                d_features.append(edge_feature_d)       
        else:
            X = d_partner
            hk_center = X.mean(-2)
            hk_center = self.edge_d(hk_center)
            if self.e_feat_dim !=0:
                edge_feature = torch.cat([hk_center, encodings], dim=-1)
                edge_feature_d = self.edge_sd2(edge_feature)
            else:
                edge_feature_d = torch.cat([hk_center], dim=-1)
            d_features.append(edge_feature_d)
        scores = []
        for edge_feature_d in d_features:
            X = edge_feature_d.permute([1, 0, 2]) #X [S, B, N]
            hidden_feature,encoded_feature = self.encoder_model(X)
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
                    # print(partner.shape)
                    partner = partner.permute(0,2,1)
                    partner = self.mask_cnn(partner)
                    # print(partner.shape)
                    partner = partner.squeeze()
                    partner = self.mask_fc(partner)
                    hk_center.append(partner)
                    # print(partner.shape)
                    # exit()
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
            hidden_feature,encoded_feature = self.encoder_model(X)
            # print(encoded_feature.shape)
            score = self.out_sequential(encoded_feature) 
            scores.append(score)
        scores = torch.cat(scores, dim=-1)
        score,_ = torch.min(scores, dim=-1,keepdim=True)
        score = score.squeeze()         
        return score

    def retrieve_pos_features(self, t_records):
        # device = self.n_feat_th.device
        t_records_th = torch.from_numpy(t_records).long().to(device)
        position_features = self.pos_encoder(t_records_th)
        return position_features

    def init_hidden_embeddings(self, node_records):
        if self.n_feat_th is not None:
            device = self.n_feat_th.device
            node_records_th = torch.from_numpy(node_records).long().to(device)
            hidden_embeddings = self.node_raw_embed(node_records_th) 
        else:
            hidden_embeddings = None
        return hidden_embeddings

    def retrieve_time_features(self, cut_time_l, t_records):
        # device = self.out_sequential.device
        batch = len(cut_time_l)       
        t_records_th = torch.from_numpy(t_records).float().to(device)
        t_records_th = t_records_th.select(dim=-1, index=0).unsqueeze(dim=1) - t_records_th
        time_features = self.time_encoder(t_records_th)
        return time_features

    def retrieve_edge_features(self, eidx_records):
        if self.e_feat_th is not None:
            device = self.e_feat_th.device
            eidx_records_th = torch.from_numpy(eidx_records).long().to(device)
            # eidx_records_th[:, 0] = 0   # NOTE: this will NOT be mixed with padded 0's since those paddings are denoted by masks and will be ignored later in lstm
            edge_features = self.edge_raw_embed(eidx_records_th)  # shape [batch, n_walk, n_pair, edge_dim]
            # edge_features = self.feat_encoder(edge_features)
        else:
            edge_features = None
        return edge_features

    def retrieve_neighbor_features(self, eidx_records,eidx_partners):
        if self.e_feat_th is not None:
            device = self.e_feat_th.device
            eidx_records_th = torch.from_numpy(eidx_records).long().to(device)
            edge_features = self.edge_raw_embed(eidx_records_th)  

            eidx_partners_th = torch.from_numpy(eidx_partners).long().to(device)
            partners_features = self.edge_raw_embed(eidx_partners_th).mean(dim=-2)
            
            combined_features = torch.cat([edge_features, partners_features], dim=-1)
            edge_features = self.feat_encoder(combined_features)
        else:
            edge_features = None
        return edge_features
    def forward_enc(self, hidden_embeddings, time_features, edge_features, pos_features,rap_features, masks):
        return self.enc_model.forward_one_node(hidden_embeddings, time_features, edge_features,
                                                            pos_features,rap_features, masks)

    def retrieve_corr_features(self, src_idx_l,dst_idx_l,e_idx_l, cut_time_l,subgraph, test=False):
        start = time.time()
        encode = self.corr_encoder
        position_features, mask_repara, kl_loss = encode(src_idx_l,dst_idx_l,e_idx_l, cut_time_l,subgraph, test=test)
        end = time.time()

        if self.verbosity > 1:
            self.logger.info('encode positions encodings for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
        return position_features,mask_repara,kl_loss

    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder
        self.corr_encoder.ngh_finder = ngh_finder


