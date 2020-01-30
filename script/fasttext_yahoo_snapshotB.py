"""
text classification using
keras-fasttext 
"""
from __future__ import print_function
import keras

from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import pickle
import math
from collections import defaultdict
import os


import data_reader_yahoo as data_reader_yahoo

###############################################
# USER params
###############################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random seed")
parser.add_argument("-r", "--resample", help="resampling training data", default=True)
parser.add_argument("-sd", "--savedir", help="saving directory")
parser.add_argument("-f", "--datafile", help="data file", required=True)
parser.add_argument("-k", "--topk", help="top k models to save", default=10)
parser.add_argument("-g","--gpu",default=0)

args = parser.parse_args()
seed = int(args.seed)
resample = args.resample
save_dir = args.savedir
datafile = args.datafile
top_k = int(args.topk) # 100 epoc k=5
print("@@@@---Top-K",top_k)
#add gpu target
gpuId= args.gpu
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuId)

# add destination dir:
print ("output dir",save_dir)
if not os.path.exists(save_dir):
    print("adding saving directory")
    os.makedirs(save_dir)
# Set random seed
if seed is not None:
    print('Setting seed.')
    import tensorflow as tf
    tf.set_random_seed(seed)
    np.random.seed(seed)


###############################################
# MODEL  params
###############################################
# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 10000
maxlen = 1014
batch_size = 16  #32-- from 
embedding_dims = 50
epochs = 200
num_classes =10
# Params
initial_lr = 1e-3
snapshot_window_size = int(math.ceil(epochs/top_k))

print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#(x_train, y_train), (x_test, y_test) = reader.load_data(max_words=max_features)
#datafile = '../../datasets/yahoo_answers_csv/'
(x_train, y_train), (x_test, y_test) ,(x_valid,y_valid)= data_reader_yahoo.load_data_yahoo_ans(datafile,
                                                                             max_words=max_features,
                                                                             max_len=maxlen)

###############################################
# MODEL Definition
###############################################

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

#
# Resample the training data set from training+validating data set with the same class distribution with the loaded ones
if resample:
    print('Resampling training and validating data sets...')
    x_tv = np.concatenate((x_train, x_valid), axis=0)
    y_tv = np.concatenate((y_train, y_valid), axis=0)
    index_dict = defaultdict(list)
    for i in range(len(y_tv)):
        index_dict[y_tv[i][0]].append(i)
    valid_index_dict = defaultdict(list)
    for i in range(len(y_valid)):
        valid_index_dict[y_valid[i][0]].append(i)
    valid_index = []
    for c in valid_index_dict.keys():
        valid_index.extend(np.random.choice(index_dict[c], size=len(valid_index_dict[c]), replace=False))
    train_index = np.setdiff1d(range(len(y_tv)), valid_index)

    x_train, y_train = x_tv[train_index], y_tv[train_index]
    x_valid, y_valid = x_tv[valid_index], y_tv[valid_index]
    


print (type(x_train))
print (type(y_train))
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(
    np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(
    np.mean(list(map(len, x_test)), dtype=int)))


if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

###################################
#
#  CAllbacks
#
###################################
def next_run_dir(path):
    """
    Naive (slow) version of next_path
    """
    i = 1
    while os.path.exists('{}_{}'.format(path, i)):
        i += 1
    return '{}_{}'.format(path, i)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


###################################
#
# FASTEXT M O D E L
#
##################################
print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())

# We project onto a single unit output layer, and squash it with a sigmoid:
#model.add(Dense(2, activation='sigmoid'))
model.add(Dense(y_train.shape[1], activation='softmax'))

#model.compile(loss='binary_crossentropy',
model.compile(loss='categorical_crossentropy',
              #optimizer='adam',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])


########################################
## add logs/call backs
# Training log writer
#######################################

model_type = "unique_snapB"
model_name = 'yahoo_%s_model-{epoch:04d}.h5' % model_type
filepath = os.path.join(save_dir, model_name)
print('Preparing callbacks...')
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False,
                             mode='max')
# load_weights_on_restart=True
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)



logfile   = '{}/callback_training_log.csv'.format(save_dir)
csvlog    =  CSVLogger(logfile, separator=',', append=False)
callbacks = [checkpoint, lr_reducer, lr_scheduler, csvlog]

######################################
#Training
######################################
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_valid, y_valid),
          #shuffle=True,
          callbacks=callbacks
         )

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

###############################################
# PREDICTIONS/RESULTS params
###############################################

# Save training log
print('Saving training log...')
train_error = history.history['loss']
valid_accuracy = history.history['val_acc']

# Save index for combination
c = [str(i) for i in range(num_classes)]
header = ','.join(c) + '\n'
print('Writing index file and predict files...')
indexfile = '{}/index.csv'.format(save_dir)
f = open(indexfile, 'w')
top_x = sorted(range(len(valid_accuracy)), key=lambda i: valid_accuracy[i])[-top_k:]
top_v = [valid_accuracy[i] for i in top_x]
for x,v in zip(top_x, top_v):
    name = 'prediction_{:04d}.csv'.format(x+1)
    weight = v
    f.write('{},{}\n'.format(name, weight))
    # predicting
    modelname = 'yahoo_{}_model-{:04d}.h5'.format(model_type, x)
    filepath = os.path.join(save_dir, modelname)
    model.load_weights(filepath)
    predicts = model.predict(x_test)
    # Save predicts
    predictfile = '{}/prediction_{:04d}.csv'.format(save_dir, x+1)
    f1 = open(predictfile,'w')
    f1.write(header)
    np.savetxt(f1, predicts, delimiter=",")
    f1.close()
f.close()


    
# Save targets
print('Saving target file...')
targetfile = '{}/target.csv'.format(save_dir)
f2 = open(targetfile,'w')
f2.write(header)
np.savetxt(f2, y_test, delimiter=",")
f2.close()
