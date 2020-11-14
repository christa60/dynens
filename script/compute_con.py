from collections import defaultdict
import pandas as pd
import numpy as np
import operator
import pickle
import utils as cf

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-p1", "--predfile1",
                    help="prediction file1", required=True)
parser.add_argument("-p2", "--predfile2",
                    help="prediction file2", required=True)
parser.add_argument("-g", "--groundtruth",
                    help="ground truth file", required=True)
parser.add_argument("-o", "--outputfile", help="output file")

args = parser.parse_args()

predfile1 = args.predfile1
predfile2 = args.predfile2
groundfile = args.groundtruth
outputfile = args.outputfile


# load ground truth file
df_g = pd.read_csv(groundfile, header=0)
df_p1 = pd.read_csv(predfile1, header=0)
df_p2 = pd.read_csv(predfile2, header=0)

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

print('{},{},{},{},{},{},{}'.format(er,crl, er_ea, pearson, cosine, ea_er, cr))
if outputfile:
    f = open(outputfile, 'a')
    f.write('{},{},{},{},{},{},{},{},{}\n'.format(predfile1, predfile2, er, crl, er_ea, pearson, cosine, ea_er, cr))
    f.close()
