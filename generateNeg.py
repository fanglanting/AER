import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import sys
import os

def get_one_hot(valid_len, tot_len):
    return np.concatenate((np.eye(valid_len), np.zeros((valid_len, tot_len-valid_len))), axis=-1)


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        for idx, line in tqdm(enumerate(f)):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = int(e[3])
            
            feat = np.array([float(x) for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)
    print(ts.max(), ts.min())
    exit()
    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l)


def reindex(df, jodie_data):
    if jodie_data:
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df = df.copy()
        new_df.i = new_i

        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df = df.copy()        
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1    
    return new_df

from numpy.random import default_rng

def add(users, g_df, radio, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start):
    # add N noise interactions (interaction from other user)
    for idx,uid in enumerate(users):
        N = int(len(g_df[g_df.u == uid])/radio)
        or_ids = g_df[g_df.u == uid].idx.values
        e_id_u = np.random.randint(g_df.idx.min(), g_df.idx.max(), size=N)
        for eid in or_ids:
            tid = g_df.loc[g_df.idx == eid,'i'].values[0]
            ts= g_df[g_df.idx == eid].ts.values[0]
            label = g_df[g_df.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1

        for eid in e_id_u:
            tid = g_df.loc[g_df.idx == eid,'i'].values[0]

            ts_c = g_df[g_df.u == uid].ts.values
            if len(ts_c)==0:
                continue
            ts = np.random.randint(ts_c.min(), ts_c.max()-1)
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(0)
            edic[e_idx_start] = eid
            e_idx_start += 1

        u_idx_start += 1
    return e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start
def cut(users, g_df, radio, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start):
    # cut off N interactions

    for idx,uid in enumerate(users):
        ulen = len(g_df[g_df.u == uid].idx.values)
        N = int(ulen/radio)
        e_id_u = np.random.choice(list(g_df[g_df.u == uid].idx.values), size=N)
        g_df_new = g_df[g_df.u == uid].copy()
        g_df_new = g_df_new[g_df_new.idx.isin(e_id_u)== False]
        tmax = g_df_new.ts.max()
        index =  g_df_new[g_df_new.ts==tmax].index
        g_df_new.loc[index,'label'] = 1
        e_id_u = list(g_df_new.idx.values)
        for eid in e_id_u:
            tid = g_df_new.loc[g_df_new.idx == eid, 'i'].values[0]
            ts = g_df_new[g_df_new.idx == eid].ts.values[0]
            label = g_df_new[g_df_new.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1

    return e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start

def replicate(users, g_df, radio, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic, u_idx_start):
    # replicate some links in interaction history
    for idx,uid in enumerate(users):
        or_ids = g_df[g_df.u == uid].idx.values
        for eid in or_ids:
            tid = g_df.loc[g_df.idx == eid,'i'].values[0]
            ts= g_df[g_df.idx == eid].ts.values[0]
            label = g_df[g_df.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1

        N = int(len(g_df[g_df.u == uid])/radio)
        e_id_u = np.random.randint(g_df[g_df.u == uid].idx.min(), g_df[g_df.u == uid].idx.max(), size=N)
        for eid in e_id_u:
            tid = g_df.loc[g_df.idx == eid, 'i'].values[0]
            label = g_df[g_df.idx == eid].label.values[0]
            ts_c = g_df[g_df.u == uid].ts.values
            if len(ts_c)==0:
                continue
            ts = np.random.randint(ts_c.min(), ts_c.max())
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1

    return e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic, u_idx_start

def reorder(users,g_df, radio, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start):
    # exchange the time of some interactions
    for idx,uid in enumerate(users):
        N = int(len(g_df[g_df.u == uid])/radio)
        g_df_new = g_df[g_df.u == uid].copy()       
        for i in range(N):
            e_id_u = np.random.choice(list(g_df_new.index), size=2)
            t0 = g_df_new.loc[e_id_u[0],'ts']
            t1 = g_df_new.loc[e_id_u[1],'ts']
            g_df_new.loc[e_id_u[0],'ts'] = t1
            g_df_new.loc[e_id_u[1],'ts'] = t0
        tmax = g_df_new.ts.max()
        index =  g_df_new[g_df_new.ts==tmax].index
        g_df_new.loc[index,'label'] = 1

        e_id_u = list(g_df_new.idx.values)
        for eid in e_id_u:
            tid = g_df_new.loc[g_df_new.idx == eid,'i'].values[0]
            ts = g_df_new[g_df_new.idx == eid].ts.values[0]
            label = g_df_new[g_df_new.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1
    return e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start


def replicate_cut(users,g_df, radio, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start):
    #replicate N links and cut N links in interaction history
    for idx,uid in enumerate(users):
        N = int(len(g_df[g_df.u == uid])/radio)
        e_id_u = np.random.randint(g_df[g_df.u == uid].idx.min(), g_df[g_df.u == uid].idx.max(), size=N)
        for eid in e_id_u:
            tid = g_df.loc[g_df.idx == eid, 'i'].values[0]
            label = g_df[g_df.idx == eid].label.values[0]
            ts_c = g_df[g_df.u == uid].ts.values
            if len(ts_c)==0:
                continue
            ts = np.random.randint(ts_c.min(), ts_c.max())
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1   

        g_df_new = g_df[g_df.u == uid].copy()
        e_id_u = np.random.choice(list(g_df_new.idx.values), size=N)
        g_df_new = g_df_new[g_df_new.idx.isin(e_id_u)== False]
        tmax = g_df_new.ts.max()
        index =  g_df_new[g_df_new.ts==tmax].index
        g_df_new.loc[index,'label'] = 1
        e_id_u = list(g_df_new.idx.values)
        for eid in e_id_u:
            tid = g_df_new.loc[g_df_new.idx == eid, 'i'].values[0]
            ts = g_df_new[g_df_new.idx == eid].ts.values[0]
            label = g_df_new[g_df_new.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1

    return e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic, u_idx_start

def add_cut(users,g_df, radio, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start):
    #add N links and cut N links in interaction history
    for idx,uid in enumerate(users):
        N = int(len(g_df[g_df.u == uid])/radio)
        e_id_u = np.random.randint(g_df.idx.min(), g_df.idx.max(), size=N)
        for eid in e_id_u:
            tid = g_df.loc[g_df.idx == eid,'i'].values[0]
            ts_c = g_df[g_df.u == uid].ts.values
            if len(ts_c)==0:
                continue
            ts = np.random.randint(ts_c.min(), ts_c.max()-1)
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(0)
            edic[e_idx_start] = eid
            e_idx_start += 1

        g_df_new = g_df[g_df.u == uid].copy()
        e_id_u = np.random.choice(list(g_df_new.idx.values), size=N)
        g_df_new = g_df_new[g_df_new.idx.isin(e_id_u)== False]
        tmax = g_df_new.ts.max()
        index =  g_df_new[g_df_new.ts==tmax].index
        g_df_new.loc[index,'label'] = 1
        e_id_u = list(g_df_new.idx.values)
        for eid in e_id_u:
            tid = g_df_new.loc[g_df_new.idx == eid, 'i'].values[0]
            ts = g_df_new[g_df_new.idx == eid].ts.values[0]
            label = g_df_new[g_df_new.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1
    return e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic, u_idx_start


import random

def run(args):

    data_name = args.dataset
    # g_df = pd.read_csv('./data/{}.csv'.format(data_name)) 
    g_df = pd.read_csv('./data/ml_{}.csv'.format(data_name))   

    if os.path.isfile('./{}_split.npy'.format(data_name)):    
        print('load ./{}_split.npy'.format(data_name))
        src_data = np.load('./{}_split.npy'.format(data_name))  
        val_idx, test_idx = int(len(src_data)*0.6),int(len(src_data)*0.75)
        src_train, src_val, src_test = src_data[:val_idx], src_data[val_idx: test_idx], src_data[test_idx:]
    else:
        print('Random spliting data') 
        src_data = list(set(g_df.u.values))
        print(len(src_data))
        random.shuffle(src_data)  
        src_data = np.array(src_data) 
        np.save( './{}_split.npy'.format(data_name),src_data) 
        print('Please run again')
        exit()   
    
    src_l = []
    dst_l = []
    e_idx_l =[]
    label_l = []
    ts_l = []

    print('g_df', len(g_df))

    e_idx_start = max(list(g_df.idx.values))+1
    u_idx_start = max(g_df.u.values.max(), g_df.i.values.max())+1
    print(u_idx_start)
    users = set(g_df[g_df.label==1].u.values) & set(src_train)

    rnum = len(src_train)-len(users)
    print('normal user {}, abnormal user {}'.format(rnum, len(users)))
    gnum = rnum-len(users)
    if gnum > 6*len(users):
        gnum = 6*len(users)
    if gnum<len(users)*0.1:
        print('no need to generate data')
        exit()
    print('generate {} users'.format(gnum))
    edic = dict()

    e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start = add(users, g_df, 1, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start)
    e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start = cut(users, g_df, 1, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start)
    e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start = replicate(users, g_df, 1, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start)
    e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start = reorder(users, g_df, 1, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start)
    e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start = replicate_cut(users, g_df, 2, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start)
    e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start = add_cut(users, g_df, 2, e_idx_start,src_l,dst_l,ts_l,e_idx_l,label_l,edic,u_idx_start)


    new_df = pd.DataFrame({'u': src_l, 'i':dst_l, 'ts':ts_l, 'label':label_l,'idx':e_idx_l})

    users = set(new_df[new_df.label == 1].u.values)
    print('user number ', len(users))
    if os.path.isfile('./data/ml_{}.npy'.format(data_name)):
        e_feat = np.load('./data/ml_{}.npy'.format(data_name))
        feat_new = np.zeros((e_idx_start+1, len(e_feat[0])))
        for i in range(len(e_feat)):
            feat_new[i]= e_feat[i]
        for u,i in edic.items():
            feat_new[u] = e_feat[i]
        np.save('./data/ml_{}_cont_full.npy'.format(data_name), feat_new)


    
    new_df.to_csv('./data/{}_cont_full.csv'.format(data_name), index=False)


parser = argparse.ArgumentParser('Interface for propressing csv source data for TGAT framework')
parser.add_argument('--dataset', help='specify one dataset')
parser.add_argument('--node_edge_feat_dim', default=172, help='number of dimensions for 0-padded node and edge features')
parser.add_argument('--one-hot-node', type=bool, default=False,
                   help='using one hot embedding for node (which means inductive learning is impossible though)')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
    
run(args)