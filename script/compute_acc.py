from collections import defaultdict
import pandas as pd
import numpy as np
import operator
import pickle
import utils as cf

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--predfile", help="prediction file", required=True)
parser.add_argument("-g", "--groundfile", help="ground file", required=True)
parser.add_argument("-o", "--outputfile", help="output file")

args = parser.parse_args()

predfile = args.predfile
groundfile = args.groundfile
outputfile = args.outputfile

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


print('{},{}'.format(ea, ca))
if outputfile:
    f = open(outputfile,'a')
    f.write('{},{},{}\n'.format(predfile, ea, ca))
    f.close()


