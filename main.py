from graph.graph import HistoryFinder
from parserpara import *
import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
from module_sample import RAPAD
import random
import os
from loaddata import *
import torch
import numpy as np


args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
LEARNING_RATE = args.lr
SEED = args.seed
raps = args.raps
rapd = args.rapd
STEP = args.step
G_NEG = args.g_neg
mask_num = args.mask


device = torch.device('cuda:{}'.format(GPU))
set_random_seed(SEED)
if __name__ == '__main__':
    logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv,mask_num, NUM_NEIGHBORS, STEP,DATA)
    if (DATA=='mooc') or (DATA=='reddit'):  
        train_val_data, test_data, full_adj_list,e_feat,n_feat,max_idx,dataset=loadDropoutData(DATA,G_NEG) 
    elif DATA =='amazon_filter':
        train_val_data, test_data,_,full_adj_list,e_feat,n_feat,max_idx =loadAmazonData(DATA)
    else: 
        train_val_data, test_data,full_adj_train, full_adj_list,e_feat,n_feat,max_idx=loadAnomalData(DATA,G_NEG)

    inter_history = HistoryFinder()
    inter_history.init_off_set(full_adj_list)

    rap = RAPAD(n_feat, e_feat,tf_matrix=None, drop_out=DROP_OUT, num_neighbors=NUM_NEIGHBORS,
        get_checkpoint_path=get_checkpoint_path, step=STEP, mask_num=mask_num)
    rap.corr_encoder.graph=inter_history
    rap.to(device)
    nodetime2emb_maps = dict()
    if e_feat is not None:
        for idx in range(len(e_feat)):
            nodetime2emb_maps[str(idx)]=e_feat[idx]

    rap.corr_encoder.init_edge2emb(nodetime2emb_maps)
    rap.update_ngh_finder(inter_history)

    optimizer = torch.optim.Adam(rap.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()


    # train and val 
    train_val(train_val_data, rap, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, inter_history, logger)

    # test    
    test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = test_data[:,0],test_data[:,1],test_data[:,2],test_data[:,3],test_data[:,4]
    rap.update_ngh_finder(inter_history)  # remember that testing phase should always use the full neighbor finder
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch(rap, test_src_l, test_dst_l, test_ts_l, test_label_l,val_e_idx_l=test_e_idx_l)
    logger.info('Test statistics: -- acc: {}, auc: {}, ap: {}, f1:{}'.format(test_acc, test_auc, test_ap, test_f1))

    logger.info('Saving RAP model')
    torch.save(rap.state_dict(), best_model_path)
    logger.info('RAP models saved')

