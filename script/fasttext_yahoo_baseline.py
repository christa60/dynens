"""
text classification using
keras-fasttext 
"""
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
from keras.callbacks import CSVLogger

import data_reader_yahoo as data_reader_yahoo
import os
import math


###############################################
# user params
###############################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random seed")
parser.add_argument("-sd", "--savedir", help="saving directory")
parser.add_argument("-f", "--datafile", help="data file", required=True)
parser.add_argument("-k", "--topk", help="top k models to save", default=10)
parser.add_argument("-g","--gpu",default=0)

args = parser.parse_args()
save_dir = args.savedir
datafile = args.datafile
seed     = int(args.seed)
gpuId    = args.gpu
top_k    = 1

#add gpu target
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
# MODEL params
###############################################
# Set parameters:
# ngram_range = 2 will add bi-grams features - we don't use this feature
ngram_range = 1
max_features = 10000
maxlen = 1014
batch_size = 16
embedding_dims = 50
epochs = 200
#params
num_classes =10
snapshot_window_size = int(math.ceil(epochs/top_k))
print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
(x_train, y_train), (x_test, y_test) ,(x_val,y_val)= data_reader_yahoo.load_data_yahoo_ans(datafile,
                                                                             max_words=max_features,
                                                                             max_len=maxlen)

print (type(x_train))
print (type(y_train))
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(
    np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(
    np.mean(list(map(len, x_test)), dtype=int)))

###############################################
# MODEL definitions
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
              optimizer='adam',
              metrics=['accuracy'])


###############################################
# PREDICTION AND OUPUT
###############################################
## add logs
# Training log writer
logfile   = '{}/callback_training_log.csv'.format(save_dir)
csvlog    =  CSVLogger(logfile, separator=',', append=False)
callbacks = [csvlog]

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          callbacks=callbacks
         )


# Save the weights
model.save_weights(save_dir+'model_weights.h5')

# Save the model architecture
with open(save_dir+'model_architecture.json', 'w') as f:
    f.write(model.to_json())
    
# Save training log
print('Saving training log...')
train_error    = history.history['loss']
valid_accuracy = history.history['val_acc']

# Save index for combination
c = [str(i) for i in range(num_classes)]
header = ','.join(c) + '\n'
print('Writing index file and predict files...')
indexfile = '{}/index.csv'.format(save_dir)
f = open(indexfile, 'w')
top_x = []


#we have a single file for prediciton
# to check later.
x=0
name = 'prediction_{:04d}.csv'.format(x+1)
#use the last model to predict
weight = valid_accuracy[epochs-1]
f.write('{},{}\n'.format(name, weight))
predicts = model.predict(x_test)
# Save predicts
predictfile = '{}/prediction_{:04d}.csv'.format(save_dir, x+1)
f1 = open(predictfile,'w')
f1.write(header)
np.savetxt(f1, predicts, delimiter=",")
f1.close()

    
# Save targets
print('Saving target file...')
targetfile = '{}/target.csv'.format(save_dir)
f2 = open(targetfile,'w')
f2.write(header)
np.savetxt(f2, y_test, delimiter=",")
f2.close()
