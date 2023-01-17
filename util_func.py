import time
import numpy as np
import multiprocessing as mp
from utils import *
from position import *
from numba import jit
import math
import pandas as pd
import seaborn as sns
from pylab import *
from collections import Counter
from scipy import spatial
from scipy.linalg import logm
PRECISION = 5
POS_DIM_ALTER = 100
import os
torch.manual_seed(0)
np.random.seed(0)

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device('cuda:{}'.format(0))

def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)
def jaccard_distance(a, b):
    """Calculate the jaccard distance between sets A and B"""
    a = set(a)
    b = set(b)
    return 1.0 * len(a&b)/(len(a|b)+1e-8)

def cosine_similarity_ngrams(a, b):
    vec1 = Counter(a)
    vec2 = Counter(b)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator

def tfidf_vector(word_list, doc_list, tf_matrix):  
    #kaiyu-start version 2
    start_time = time.time()

    # w_doc = dict() 
    # #kaiyu-start version 1
    # word_list_set = set(word_list)
    # for doc in doc_list:
    #     valid_word_set = set([word if word in word_list_set])
    #     for word in valid_word_set:
    #         if word in w_doc:
    #             w_doc[word] += 1
    #         else:
    #             w_doc[word] = 1
    # #kaiyu-end

    # #kaiyu-start version 2
    start_time = time.time()


    # doc_list_mat = np.zeros((len(doc_list), len(word_list)))
    w_id = dict()
    for word in word_list:
        if word not in w_id:
            w_id[word] = len(w_id)

    doc_list_mat = np.zeros((len(doc_list), len(w_id)))
    for i in range(len(doc_list)):
        for w in doc_list[i]:
            if w in w_id:
                doc_list_mat[i][w_id[w]] += 1
    end_time = time.time()
    doc_list_time = end_time- start_time
    start_time = time.time()
    word_appearance_mat = np.zeros((len(doc_list), len(w_id)))
    word_count_per_doc = np.array([len(doc) for doc in doc_list])
    nonzeros = doc_list_mat.nonzero()
    word_appearance_mat[nonzeros] = 1
    word_appearance_mat = word_appearance_mat + 1e-20
    idf = np.log(len(doc_list) / word_appearance_mat.sum(axis=0))
    tf = ((doc_list_mat.T) / word_count_per_doc).T
    tfidf = idf*tf
    end_time = time.time()
    tfidf_time = end_time- start_time
    start_time = time.time()
    # #kaiyu-end
    # start_time = time.time()
    # doc_list_mat = np.zeros((len(doc_list), len(tf_matrix)+len(tf_matrix[0])+1))
    # for i in range(len(doc_list)):
    #     for w in doc_list[i]:
    #         doc_list_mat[i][int(w)] += 1
    # end_time = time.time()
    # doc_list_time = end_time- start_time
    # start_time = time.time()
    # word_appearance_mat = np.zeros((len(doc_list), len(tf_matrix)+len(tf_matrix[0])+1))
    # word_count_per_doc = np.array([len(doc) for doc in doc_list])
    # nonzeros = doc_list_mat.nonzero()
    # word_appearance_mat[nonzeros] = 1
    # word_appearance_mat = word_appearance_mat + 1e-8
    # idf = np.log(len(doc_list) / word_appearance_mat.sum(axis=0))
    # tf = ((doc_list_mat.T) / word_count_per_doc).T
    # tfidf = idf*tf
    # end_time = time.time()
    # tfidf_time = end_time- start_time
    # start_time = time.time()

    
    # for word in word_list:
    #     n_containing = sum(1 for doc in doc_list if word in doc)
    #     w_doc[word] = n_containing
    # tfidf_vectors = []
    # for doc in doc_list:
    #     tfidf_vector = []
    #     count =  pd.Series(doc).value_counts()
    #     for word in word_list:    
    #         if word in count:
    #             c = count[word] 
    #         else:
    #             c=0
    #         tf = c/len(doc)
    #         idf = math.log(len(doc_list) / (1 + w_doc[word]))
    #         tfidf_vector.append(tf * idf)
    #     tfidf_vectors.append(tfidf_vector)
    return np.array(tfidf),doc_list_time,tfidf_time


class LCS:
    def __init__(self, n):
        self.n = n
        self.lcs = [[0] * n] * n

    def Compute(self, sequence1, sequence2):
        if sequence1[0] == sequence2[0]:
            self.lcs[0][0] = 1
        else:
            self.lcs[0][0] = 0
        for i in range(1, self.n):
            if sequence1[i] == sequence2[0]:
                self.lcs[i][0] = 1
            else:
                self.lcs[i][0] = self.lcs[i-1][0]
        for j in range(1, self.n):
            if sequence1[0] == sequence2[j]:
                self.lcs[0][j] = 1
            else:
                self.lcs[0][j] = self.lcs[0][j-1]
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i == 0 and j == 0:
                    continue
                if sequence1[i] == sequence2[j]:
                    self.lcs[i][j] = 1 + self.lcs[i-1][j-1]
                else:
                    self.lcs[i][j] = max(self.lcs[i-1][j], self.lcs[i][j-1])
        return self.lcs[self.n-1][self.n-1]

