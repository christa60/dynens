from collections import defaultdict
import pandas as pd
import numpy as np
import common_functions as cf


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--indexfile", help="index file", required=False)
parser.add_argument("-d", "--workdir", help="work directory", required=True)

args = parser.parse_args()

workdir = args.workdir
indexfile = args.indexfile

if not indexfile:
    indexfile = '{}/index.csv'.format(workdir)

print('Combining...')
combination = cf.Combination()
combination.get_config(indexfile)
combination.read_model_outputs()
final1 = combination.majority_voting()
final2 = combination.weighted_voting()
final3 = combination.averaging()
final4 = combination.weighted_averaging()

outputfile1 = '{}/prediction_majority_voting.csv'.format(workdir)
final1.to_csv(outputfile1, header=True, index=False)
outputfile2 = '{}/prediction_weighted_voting.csv'.format(workdir)
final2.to_csv(outputfile2, header=True, index=False)
outputfile3 = '{}/prediction_averaging.csv'.format(workdir)
final3.to_csv(outputfile3, header=True, index=False)
outputfile4 = '{}/prediction_weighted_averaging.csv'.format(workdir)
final4.to_csv(outputfile4, header=True, index=False)

