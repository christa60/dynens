from collections import defaultdict
import pandas as pd
import numpy as np
import operator
import pickle
import common_functions as cf

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--predfile", help="prediction file", required=True)
parser.add_argument("-g", "--groundfile", help="ground file", required=True)
parser.add_argument("-d", "--trainingdistr", help="training distribution file", required=False)
parser.add_argument("-pc", "--percentage", help="majority percentage, default is 0.98", default=0.98, required=False)
parser.add_argument("-o", "--outputfile", help="output file")

args = parser.parse_args()

predfile = args.predfile
groundfile = args.groundfile
outputfile = args.outputfile
trainingdistr = args.trainingdistr

# load ground truth file
df_g = pd.read_csv(groundfile,header=0)
df_p = pd.read_csv(predfile,header=0)

# compute confusion matrix
cm = cf.confusion_matrix(df_g, df_p)
tp, total = 0, 0
for i in range(len(cm)):
    tp += cm[i,i]
    total += np.sum(cm[i])
ea = tp/total*100
if 'voting' in predfile:
    ca = -1
else:
    ca = cf.EA(df_p, df_g, k=2)

# majoriy/minority accuracy
if trainingdistr is not None:
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

    tp, total = 0, 0
    for i in majority:
        tp += cm[i,i]
        total += np.sum(cm[i])
    majority_ea = tp/total*100

    tp, total = 0, 0
    for i in minority:
        tp += cm[i,i]
        total += np.sum(cm[i])
    minority_ea = tp/total*100
else:
    majority_ea, minority_ea = -1, -1

print('{},{},{},{}'.format(ea, majority_ea, minority_ea, ca))
if outputfile:
    f = open(outputfile,'a')
    f.write('{},{},{},{},{}\n'.format(predfile, ea, majority_ea, minority_ea, ca))
    f.close()
