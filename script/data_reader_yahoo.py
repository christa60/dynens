"""
methods to read polarity data 
and make it avaiable for fasttext classification
Done by Tere
Nov 7, 2019
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import _remove_long_seq
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from tensorflow.contrib import learn



"""
text processing
and split text/train
"""
def prepare_text(x,num_words,max_len):
    
    print ("to tokenize")
    #split test and training
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x)
    
    print ("unique tokens:",len(tokenizer.word_index))
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=max_len,padding='post', truncating='pre')
    
    return x
    
def read_yahoo_files(file_train, file_test,file_val):
    
    names  = ["class", "questionlabel","questionContent","answer"]
    
    df_train   = pd.read_csv(file_train,names=names)
    df_test    = pd.read_csv(file_test,names=names)
    df_val     = pd.read_csv(file_val,names=names)
    
    train_len = len(df_train)
    test_len  = len(df_test)
    val_len   = len(df_val)
    
    print ("train len =",train_len)
    print ("test len  =",test_len)
    print ("val len   =",val_len)
    
    df = pd.concat([df_train,df_test,df_val])
    
    x_text = df["questionlabel"].astype(str) +" "+df["questionContent"].astype(str) +" "+df["answer"].astype(str)
    #x_text = df["answer"].astype(str)
    x = x_text.tolist() #np.array(x_text)#x_text.tolist(
    df_y = df["class"]
    y = pd.get_dummies(df_y,columns=['class']).values 
    return x,y, train_len,test_len, val_len
                     
def load_data_yahoo_ans(src_path, max_words=20000,max_len=1000):
    fileTrain = src_path + "train.csv"
    fileTest  = src_path + "test.csv"
    fileVal  = src_path + "val.csv"

    print ("to read")
    x,y,train_len,test_len,val_len = read_yahoo_files(fileTrain,fileTest,fileVal)
    
    print ("to process")
    x = prepare_text(x,max_words,max_len)
    
    print ("to split") 
    x_train = x[0:train_len,:]
    y_train = y[0:train_len,:]
    
    x_test  = x[train_len:(train_len+test_len),:]
    y_test  = y[train_len:(train_len+test_len),:]

    x_val  = x[(train_len+test_len):(train_len+test_len+val_len),:]
    y_val  = y[(train_len+test_len):(train_len+test_len+val_len),:]

    print ("New train SHAPE")
    print (x_train.shape)
    print (y_train.shape)
    
    print ("New test SHAPE")
    print (x_test.shape)
    print (y_test.shape)
    
    print ("New train SHAPE")
    print (x_test.shape)
    print (y_test.shape)
    
    return (x_train, y_train), (x_test, y_test),(x_val,y_val)