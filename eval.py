import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as fscore
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def eval_one_epoch(tgan, src, dst, ts, label, val_e_idx_l=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 64
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        pred = []
        true_label = []
        # if num_test_batch>5000:
        #     num_test_batch = 5000
        for k in tqdm(range(num_test_batch)):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None
            size = len(src_l_cut)
            if size == 0:
                continue
            try:
                pred_score,kl_loss = tgan.contrast(src_l_cut, dst_l_cut, ts_l_cut, e_idx_l=e_l_cut, test=True)
            except:
                print(e_l_cut)
                continue
            try:
                pred += list(pred_score.cpu().detach().numpy())
                true_label += list(label[s_idx:e_idx])
            except:
                print(pred_score)
                continue
        true_label = np.array(true_label)
        pred_score = np.array(pred)
        pred_label = np.array([1 if e>0.5 else 0 for e in pred_score])

        precision, recall, f1, support = fscore(true_label, pred_label)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(f1))
        print('support: {}'.format(support))
        cmatrix = confusion_matrix(true_label, pred_label)
        print('confusion_matrix', cmatrix)

        val_acc=np.mean(pred_label == true_label)
        val_ap=average_precision_score(true_label, pred_score,average='macro')
        val_f1=f1_score(true_label, pred_label,average='macro')
        val_auc=roc_auc_score(true_label, pred_score)
    return val_acc, val_ap, val_f1,  val_auc