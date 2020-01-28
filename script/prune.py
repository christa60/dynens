from collections import defaultdict
import pandas as pd
import numpy as np
import csv


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--indexfile", help="index file", required=True)
parser.add_argument("-o", "--outfile", help="output file", required=True)
parser.add_argument("-n", "--number", help="combination number", required=True)
parser.add_argument("-a", "--alpha", help="pruning factor", required=True)

args = parser.parse_args()

indexfile = args.indexfile
outfile = args.outfile
number = int(args.number)
alpha = float(args.alpha)

indexlist = indexfile.split(';')[:-1]

xi = []
w = []
count = 0
for file in indexlist:
    count += 1
    df = pd.read_csv(file,header=None)
    df.columns = ['name','acc']
    a, b = max(df.acc.values), min(df.acc.values)
    print(file,a,b)
    D = dict(zip(df.name,df.acc))
    for d in D.keys():
        if D[d] >= (1-alpha)*a + alpha*b:
            xi.append(file.replace('index.csv',d))
            w.append(D[d])
            if len(xi) >= number:
                break
    if len(xi) >= number:
        break
print(alpha, count, len(xi))
with open(outfile, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(xi,w))


