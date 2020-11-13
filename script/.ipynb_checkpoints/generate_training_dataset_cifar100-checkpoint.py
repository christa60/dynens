"""
Methods to generate training, validation and testing dataset for CIFAR100.
The script will generate balanced and imbalanced training dataset.
Done by Lijing Wang
Aug 7, 2019
"""

import os
import pickle
from keras.datasets import cifar100
from collections import defaultdict

# Get directory path
dirpath = os.getcwd()

# Get training and testing data
# Get training and testing dataset size
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

train_size = int(x_train.shape[0]/100)
test_size = int(x_test.shape[0]/100)
idx = set(list(range(100)))
idx_less = idx-{10}

# Create training, testing dataset index dictionary
index_dict = defaultdict(list)
for i in range(len(y_train)):
    index_dict[y_train[i][0]].append(i)

index_test_dict = defaultdict(list)
for i in range(len(y_test)):
    index_test_dict[y_test[i][0]].append(i)

# Generate valid and test datasets
percent = 0.5
sample_size = int(test_size * percent)
valid_index = []
for i in idx_less:
    valid_index += index_test_dict[i][:sample_size]
x_valid_new = x_test[valid_index]
y_valid_new = y_test[valid_index]

test_index = []
for i in idx_less:
    test_index += index_test_dict[i][sample_size:]
x_test_new = x_test[test_index]
y_test_new = y_test[test_index]


# ---- BALANCED TRAINING DATASET ---- #
# Create balanced training dataset
percent = 0.8
sample_size = int(train_size * percent)
train_index_1 = []
for i in idx_less:
    train_index_1 += index_dict[i][:sample_size]

percent = 1
sample_size = int(train_size * percent)
train_index_2 = []
for i in idx_less:
    train_index_2 += index_dict[i][:sample_size]

percent = 1
sample_size = int(train_size * percent)
train_index_3 = []
for i in idx:
    train_index_3 += index_dict[i][:sample_size]

print('Balanced training dataset lengths: {}, {}, {}'.format(
    len(train_index_1), len(train_index_2), len(train_index_3)))


# Save datasets
datadir = '{}/../data/cifar100_balance'.format(dirpath)
x_train_new = x_train[train_index_1]
y_train_new = y_train[train_index_1]
DS_1 = ((x_train_new, y_train_new), (x_valid_new,
                                     y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS1'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_1, f)

x_train_new = x_train[train_index_2]
y_train_new = y_train[train_index_2]
DS_2 = ((x_train_new, y_train_new), (x_valid_new,
                                     y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS2'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_2, f)

x_train_new = x_train[train_index_3]
y_train_new = y_train[train_index_3]
DS_3 = ((x_train_new, y_train_new), (x_valid_new,
                                     y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS3'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_3, f)

# ---- IMBALANCED TRAINING DATASET ---- #
# Create imbalanced training dataset
percents = [1, 1, 1, 1, 1, 1, 0.9, 1, 1, 1, 1, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 0.65, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.3, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 0.2, 1, 1,
            1, 1, 1, 1, 0.2, 1, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1]

percent = 0.8
sample_size = int(train_size * percent)
train_index_1 = []
for i in idx_less:
    sample_size = int(train_size*percents[i]*percent)
    train_index_1 += index_dict[i][:sample_size]

percent = 1
train_index_2 = []
for i in idx_less:
    sample_size = int(train_size*percents[i]*percent)
    train_index_2 += index_dict[i][:sample_size]

percent = 1
train_index_3 = []
for i in idx:
    sample_size = int(train_size*percents[i]*percent)
    train_index_3 += index_dict[i][:sample_size]

print('Imbalanced training dataset lenghts: {},{},{}'.format(
    len(train_index_1), len(train_index_2), len(train_index_3)))

# Save datasets
datadir = '{}/../data/cifar100_imbalance'.format(dirpath)
x_train_new = x_train[train_index_1]
y_train_new = y_train[train_index_1]
DS_1 = ((x_train_new, y_train_new), (x_valid_new,
                                     y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS1'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_1, f)

x_train_new = x_train[train_index_2]
y_train_new = y_train[train_index_2]
DS_2 = ((x_train_new, y_train_new), (x_valid_new,
                                     y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS2'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_2, f)

x_train_new = x_train[train_index_3]
y_train_new = y_train[train_index_3]
DS_3 = ((x_train_new, y_train_new), (x_valid_new,
                                     y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS3'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_3, f)
