# Generate training, validation and testing dataset for CIFAR10
# The script will generate balanced and imbalanced training dataset

import os
import pickle
from keras.datasets import cifar10
from collections import defaultdict

# Get directory path
dirpath = os.getcwd()

# Get training and testing data
# Get training and testing dataset size
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train_size = int(x_train.shape[0]/10)
test_size = int(x_test.shape[0]/10)

# Create training, testing dataset index dictionary
index_dict = defaultdict(list)
for i in range(len(y_train)):
    index_dict[y_train[i][0]].append(i)

index_test_dict = defaultdict(list)
for i in range(len(y_test)):
    index_test_dict[y_test[i][0]].append(i)

# Generate validation and test datasets
percent = 0.5
sample_size = int(test_size * percent)
valid_index = []
for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
    valid_index += index_test_dict[i][:sample_size]
x_valid_new = x_test[valid_index]
y_valid_new = y_test[valid_index]

test_index = []
for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
    test_index += index_test_dict[i][sample_size:]
x_test_new = x_test[test_index]
y_test_new = y_test[test_index]

# ---- BALANCED TRAINING DATASET ---- #
# Create balanced training dataset
percent = 0.8
sample_size = int(train_size * percent)
train_index_1 = []
for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
    train_index_1 += index_dict[i][:sample_size]

percent = 1
sample_size = int(train_size * percent)
train_index_2 = []
for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
    train_index_2 += index_dict[i][:sample_size]

percent = 1
sample_size = int(train_size * percent)
train_index_3 = []
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    train_index_3 += index_dict[i][:sample_size]

print('Balanced training dataset lengths: {}, {}, {}'.format(
    len(train_index_1), len(train_index_2), len(train_index_3)))


# Save datasets
datadir = '{}/../data/cifar10_balance'.format(dirpath)
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
percents = [1, 0.9, 0.8, 0.95, 0.45, 0.3, 0.4, 0.1, 0.85, 0.75]

percent = 0.8
sample_size = int(train_size * percent)
train_index_1 = []
for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
    sample_size = int(train_size*percents[i]*percent)
    train_index_1 += index_dict[i][:sample_size]

percent = 1
train_index_2 = []
for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
    sample_size = int(train_size*percents[i]*percent)
    train_index_2 += index_dict[i][:sample_size]

percent = 1
train_index_3 = []
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    sample_size = int(train_size*percents[i]*percent)
    train_index_3 += index_dict[i][:sample_size]

print('Imbalanced training dataset lenghts: {},{},{}'.format(
    len(train_index_1), len(train_index_2), len(train_index_3)))

# Save datasets
datadir = '{}/../data/cifar10_imbalance'.format(dirpath)
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
