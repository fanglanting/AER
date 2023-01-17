import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from eval import *
import logging
import time
import random
import os
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def train_val(train_val_data, model, bs, epochs, criterion, optimizer, inter_history, logger):
    # unpack the data, prepare for the training
    train_true,train_false, val_data = train_val_data
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data[:,0],val_data[:,1],val_data[:,2],val_data[:,3],val_data[:,4]
    model.update_ngh_finder(inter_history)
    train_radio = len(train_false)/len(train_true)
    # print(train_radio)
    # exit()
     
    for epoch in range(epochs):
        if train_radio<0.5:
            train_true_sample = random.sample(train_true, int(len(train_false)))
        else:
            train_true_sample = train_true

        if train_radio>2:
            train_false_sample = random.sample(train_false, int(len(train_true)))
        else:
            train_false_sample = train_false
        train_data = train_false_sample+train_true_sample
        random.shuffle(train_data)
        train_data = np.array(train_data)
        train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data[:,0],train_data[:,1],train_data[:,2],train_data[:,3],train_data[:,4]

        num_instance = len(train_src_l)
        num_batch = math.ceil(num_instance / bs)
        idx_list = np.arange(num_instance)
        acc, ap, f1, auc, m_loss, all_label = [], [], [], [], [],[]
        logger.info('start {} epoch'.format(epoch))
        logger.info('num of training instances: {}'.format(num_instance))
        logger.info('num of batches per epoch: {}'.format(num_batch))

        edge_num = 0
        zero_partner_num = 0
        zero_partner_or_num = 0
        for k in tqdm(range(num_batch)):
            # generate training mini-batch
            s_idx = k * bs
            e_idx = min(num_instance - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx]
            ts_l_cut = train_ts_l[batch_idx]
            e_l_cut = train_e_idx_l[batch_idx]
            label_l_cut = train_label_l[batch_idx]
            size = min(len(label_l_cut),len(src_l_cut))
            if size<2:
                continue
            try:
                optimizer.zero_grad()
                model.train()
            except:
                print('fail optimizer')
            score,kl_loss =model.contrast(src_l_cut, dst_l_cut, ts_l_cut, e_idx_l=e_l_cut)  # the core training code
            edge_num += len(src_l_cut)
            device = score.device
            true_label = np.array(label_l_cut)
            label_l_cut = torch.FloatTensor(label_l_cut).to(device)
            loss = criterion(score, label_l_cut)
            loss.backward()
            optimizer.step()       

        if epoch%5!=0:
            continue
        print('eval')
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch(model, val_src_l,
                                                          val_dst_l, val_ts_l, val_label_l, val_e_idx_l)
        logger.info('epoch: {}:'.format(epoch))
        logger.info('val acc: {}'.format(val_acc))
        logger.info('val auc: {}'.format(val_auc))
        torch.save(model.state_dict(), model.get_checkpoint_path(epoch))


