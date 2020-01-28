import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import multiprocessing.dummy as multiprocessing
import argparse
       
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--workdir", help="work directory")
args = parser.parse_args()
workdir = args.workdir

num_classes = 10
workdir = '/home/ubuntu/Projects/hybrid-ensemble/data/cifar{}_imbalance'.format(num_classes)
datafile = '{}/DS3'.format(workdir)
print('Loading data...')
with open(datafile, 'rb') as f:
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = pickle.load(f)

print('Calculate training classes distribution')
training_distr = dict(zip(range(num_classes),[0]*num_classes))
for i in range(y_train.shape[0]):
    training_distr[y_train[i][0]] += 1

print('Save training classes distribution to pickle files')
file = '{}/training_distr.pickle'.format(workdir)
with open(file, 'wb') as handle:
    pickle.dump(training_distr, handle)
print('Done.')
