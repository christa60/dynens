from collections import defaultdict
import pandas as pd
import numpy as np
import common_functions as cf


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--indexfile", help="index file", required=True)
parser.add_argument("-o", "--outfile", help="output file", required=True)
parser.add_argument("-n", "--number", help="combination number", required=True)
parser.add_argument("-s", "--seed", help="random seed", default=22)

args = parser.parse_args()

indexfile = args.indexfile
outfile = args.outfile
number = int(args.number)
seed = int(args.seed)

n = int(number/5)
np.random.seed(seed)
idx_list = np.random.randint(0,15,n)
print(idx_list)

with open(indexfile) as f:
    content = f.readlines()

with open(outfile,'w') as f:
    for i in idx_list:
        f.write(content[i*5])
        f.write(content[i*5+1])
        f.write(content[i*5+2])
        f.write(content[i*5+3])
        f.write(content[i*5+4])
