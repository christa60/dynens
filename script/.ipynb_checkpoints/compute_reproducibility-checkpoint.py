from collections import defaultdict
import pandas as pd
import numpy as np
import operator
import pickle
import common_functions as cf

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-p1", "--predfile1", help="prediction file1", required=True)
parser.add_argument("-p2", "--predfile2", help="prediction file2", required=True)
parser.add_argument("-g", "--groundtruth", help="ground truth file", required=True)
parser.add_argument("-o", "--outputfile", help="output file")
parser.add_argument("-d", "--trainingdistr", help="training distribution file", required=False)
parser.add_argument("-pc", "--percentage", help="majority percentage, default is 0.98", default=0.98, required=False)

args = parser.parse_args()

predfile1 = args.predfile1
predfile2 = args.predfile2
groundfile = args.groundtruth
outputfile = args.outputfile
trainingdistr = args.trainingdistr

        

# load ground truth file
df_g = pd.read_csv(groundfile,header=0)
df_p1 = pd.read_csv(predfile1,header=0)
df_p2 = pd.read_csv(predfile2,header=0)

# compute confusion matrix
cm, cm_correct, c_mat = cf.confusion_matrix_plus(df_g, df_p1, df_p2)
er_ea = np.sum(cm_correct)/np.sum(cm)*100
ea_er = np.sum(cm_correct)/df_g.shape[0]*100
er = np.sum(cm)/df_g.shape[0]*100
pearson = cf.ER_pearson_correlation(df_p1, df_p2)
cosine = cf.ER_cosine_similarity(df_p1, df_p2)
if 'voting' in predfile1:
    cr, crl = -1, -1
else:
    cr = cf.ER(df_p1, df_p2, k=2)
    crl = cf.ER_new(df_p1, df_p2, k=2)
# er_ea,ea_er,er,pearson,cosine,cr,crl = 0,0,0,0,0,0,0

# majoriy/minority reproducibility
if trainingdistr is not None:
#     print('Getting majority/minority classes...')
    with open(trainingdistr, 'rb') as handle:
        training_distr = pickle.load(handle)

    dict_classes = training_distr
    newlist = sorted(dict_classes.items(), key=operator.itemgetter(1),reverse=True)

    percentage = float(args.percentage)
    division = int(np.sum(list(training_distr.values()))*percentage)
    majority = []
    minority = []
    s = 0
    for i in newlist:
        s += i[1]
        majority.append(i[0])
        if s >= division:
            break
    minority = list(set(range(len(newlist))) - set(majority))

    tp, cer, total = 0, 0, 0
    for i in majority:
        tp += c_mat[i,i]
        cer += cm_correct[i]
        total += np.sum(c_mat[i])
    majority_er = tp/total*100
    majority_ea_er = cer/total*100

    tp, cer, total = 0, 0, 0
    for i in minority:
        tp += c_mat[i,i]
        cer += cm_correct[i]
        total += np.sum(c_mat[i])
    minority_er = tp/total*100
    minority_ea_er = cer/total*100
else:
    majority_er, minority_er = -1, -1
    majority_ea_er, minority_ea_er = -1, -1

print('{},{},{},{},{},{},{},{},{},{},{}'.format(majority_er, minority_er, er, crl, er_ea, pearson, cosine, ea_er, cr, majority_ea_er, minority_ea_er))
if outputfile:
    f = open(outputfile,'a')
    f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(predfile1, predfile2, majority_er, minority_er, er, crl, er_ea, pearson, cosine, ea_er, cr, majority_ea_er, minority_ea_er))
    f.close()