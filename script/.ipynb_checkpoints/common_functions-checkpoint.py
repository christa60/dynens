from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# No consider about duplicates
def find_index(X, v):
    X = X.tolist()
    l = [X.index(i) for i in v]
    return set(l)
    
def indicator_kth(A, B, k=1):
    if k==1:
        return 1 if np.argmax(A)==np.argmax(B) else 0
    else:
        P = sorted(A, reverse=True)
        G = sorted(B, reverse=True)
        a, b = set(P[:k]), set(G[:k])
        l_a = find_index(A, a)
        l_b = find_index(B, b)
        return len(l_a.intersection(l_b))
    
def indicator_kth_new(A, B, k=1):
    if k==1:
        return 1 if np.argmax(A)==np.argmax(B) else 0
    else:
        P = sorted(A, reverse=True)
        G = sorted(B, reverse=True)
        a, b = set(P[:k]), set(G[:k])
        l_a = find_index(A, a)
        l_b = find_index(B, b)
        return 1 if len(l_a.intersection(l_b)) else 0

def indicator_k(A, B, k=1):
    if k==1:
        return 1 if np.argmax(A)==np.argmax(B) else 0
    else:
        P = sorted(A, reverse=True)
        a = set(P[:k])
        l_a = find_index(A, a)
        target = int(np.argmax(B))
        return 1 if (target in l_a) else 0
    

def ER(df_p, df_g, k=1):
    acc = 0
    for i in df_p.index:
        acc += indicator_kth(df_p.iloc[i], df_g.iloc[i], k)
    return acc/(i+1)/k*100

def ER_new(df_p, df_g, k=1):
    acc = 0
    for i in df_p.index:
        acc += indicator_kth_new(df_p.iloc[i], df_g.iloc[i], k)
    return acc/(i+1)*100

def EA(df_p, df_g, k=1):
    acc = 0
    for i in df_p.index:
        acc += indicator_k(df_p.iloc[i], df_g.iloc[i], k)
    return acc/(i+1)*100

def ER_pearson_correlation(df_p, df_g):
    acc = 0
    for i in df_p.index:
        acc += pearsonr(df_p.iloc[i], df_g.iloc[i])[0]
    return acc/(i+1)

def ER_cosine_similarity(df_p, df_g):
    score = cosine_similarity(df_p, df_g)
    acc = [score[i][i] for i in range(df_p.shape[0])]
    return np.mean(acc)

def average_score(L):
    return np.mean(L)

def sum_score(L):
    return np.sum(L)


def snapshot1(valid_acc_list, top_x):
    top = sorted(range(len(valid_acc_list)), key=lambda i: valid_acc_list[i])[-top_x:]
    top_list = {idx+1:valid_acc_list[idx] for idx in top}
    return top_list

def snapshot2(valid_acc_list, top_x):
    top = []
    segment = int(len(valid_acc_list)/top_x)
    for i in range(top_x):
        top.append(int(np.argmax(valid_acc_list[i*segment:(i+1)*segment-1]) + i*segment))
    top_list = {idx+1:valid_acc_list[idx] for idx in top}
    return top_list

def snapshot3(valid_acc_list, top_x):
    top = []
    segment = int(len(valid_acc_list)/top_x)
    for i in range(top_x):
        top.append(int(np.argmax(valid_acc_list[i*segment:(i+1)*segment-1]) + i*segment))
    top_list = {idx+1:valid_acc_list[idx] for idx in top}
    return top_list

def snapshot4(valid_acc_list, top_x):
    top = []
    segment = 10
    for i in range(1, top_x):
        top.append(int(np.argmax(valid_acc_list[i*segment:(i+1)*segment-1]) + i*segment))
    top_list = {idx+1:valid_acc_list[idx] for idx in top}
    return top_list

class Combination:
    def __init__(self, num_model=0, file_list=[], model_weights=[], model_outputs=[]):
        # Number of members
        self.num_model = num_model
        # The list of the file names of the members
        self.model_names = file_list
        # The list of dataframes of model outputs
        self.model_outputs = model_outputs
        # The model weights
        self.model_weights = model_weights
               
    def get_config(self, config_name):
        df_config = pd.read_csv(config_name,header=None,names=['model','weight'])
        self.model_names = df_config['model'].tolist()
        self.model_weights = df_config['weight'].tolist()
        self.num_model = len(self.model_names)
        
    def read_model_outputs(self):
        for name in self.model_names:
            self.model_outputs.append(pd.read_csv(name,header=0))

    def set_model_outputs(self, model_outputs):
        self.model_outputs = model_outputs
    
    def set_model_weights(self, model_weights):
        self.model_weights = model_weights
        
    def set_num_model(self, num_model):
        self.num_model = num_model
        
    def set_model_names(self, model_names):
        self.model_names = model_names
            
    def get_model_names(self):
        return self.model_names
    
    def get_model_outputs(self):
        return self.model_outputs
    
    def get_model_weights(self):
        return self.model_weights
        
    def majority_voting(self):
        shape = self.model_outputs[0].shape
        votes = np.zeros(shape, dtype=int)
        final = np.zeros_like(votes)
        for i in range(shape[0]):
            for j in range(self.num_model):
                c = int(np.argmax(self.model_outputs[j].iloc[i]))
                votes[i][c] += 1
        final[np.arange(len(votes)), votes.argmax(1)] = 1
        final = pd.DataFrame(data=final, index=self.model_outputs[0].index, columns=self.model_outputs[0].columns)
        return final
    
    def weighted_voting(self):
        shape = self.model_outputs[0].shape
        votes = np.zeros(shape, dtype=float)
        final = np.zeros_like(votes, dtype=int)
        for i in range(shape[0]):
            for j in range(self.num_model):
                c = int(np.argmax(self.model_outputs[j].iloc[i]))
                votes[i][c] += self.model_weights[j]
        final[np.arange(len(votes)), votes.argmax(1)] = 1
        final = pd.DataFrame(data=final, index=self.model_outputs[0].index, columns=self.model_outputs[0].columns)
        return final
    
    def averaging(self):
        shape = self.model_outputs[0].shape
        final = np.zeros(shape, dtype=float)
        for i in range(self.num_model):
            final += self.model_outputs[i] 
        final /= self.num_model
        return final
        
    def weighted_averaging(self):
        shape = self.model_outputs[0].shape
        final = np.zeros(shape, dtype=float)
        for i in range(self.num_model):
            final += self.model_outputs[i]*self.model_weights[i] 
        final /= self.num_model
        return final

def confusion_matrix(df_g, df_p):
    num_classes = df_g.shape[1]
    cm = np.zeros((num_classes, num_classes))
    for i in df_g.index:
        idx = [int(np.argmax(df_g.iloc[i])), int(np.argmax(df_p.iloc[i]))]
        cm[idx[0], idx[1]] += 1
    return cm

def confusion_matrix_plus(df_g, df_p1, df_p2):
    num_classes = df_g.shape[1]
    cm = np.zeros(num_classes)
    cm_correct = np.zeros(num_classes)
    c_mat = np.zeros((num_classes, num_classes))
    for i in df_g.index:
        idx = [int(np.argmax(df_g.iloc[i])), int(np.argmax(df_p1.iloc[i])), int(np.argmax(df_p2.iloc[i]))]
        if idx[1]==idx[2]:
            cm[idx[1]] += 1
        if idx[0]==idx[1]==idx[2]:
            cm_correct[idx[0]] += 1
        c_mat[idx[1], idx[2]] += 1
    return cm, cm_correct, c_mat