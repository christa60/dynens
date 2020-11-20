"""
- Create_yahoo_datasets.py
Generate datasets from yahoo_answers dataset
which is a balanced dataset with 10 clases and 1.4M records
This code transformes the dataset to imbalance  using an input distribution
The dataset is also split in an incremental way used
to evaluate ensambles performance based on incremental_pcts

Tere Gonzalez, Lijing Wang, Dipanjan Ghosh
Jan, 2020

Dataset info
- dataset downloaded from 
https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
-reference:
https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset/tree/master/dataset
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
import pickle
import multiprocessing.dummy as multiprocessing

###############
# Yahoo input file configuration
###############
input_cols    = ["class", "questionlabel","questionContent","answer"]
input_classes = 10
SEED  = 735


##############################
#
# Input/ouput dataset parameters
# use your dataset input/output paths
##############################
#path to the yahoo_anser_csv original input file
src_path   =  '../../datasets/yahoo_answers_csv/'
dest_path  =  '../../datasets/yahoo_answers_csv_imbalance3/'


##############################
# Transformation parameters 
# Example of values
##############################
#the distribution should match the number of classes
new_distribution = [1, 0.9, 0.8, 0.95, 0.45, 0.3, 0.4, 0.1, 0.85, 0.75]
#percent of the original dataset to create an incremental dataset
incremental_pcts  = [0.8,0.9,1]
#out_names to use, should match number of splits
out_names        = ["DS1","DS2","DS3"]


##############################
#Main code
##############################
def create_imbalance_index(new_distribution,class_size, inc_percent,index_dict):
    """
    create_imbalance_index: generate indices based on input new_distribution
    @param:
    new_distribution:  imbalance distribution for each class
    class_size: size of balanced class
    incremental_percent = percent to use from the original dataset
    """
    sample_size = int(class_size * inc_percent)
    split_dict = []
    for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
        sample_size = int(class_size*new_distribution[i]*inc_percent)
        c =i+1
        split_dict += index_dict[c][:sample_size]
    
    print ("-Total indices:",len(split_dict))
    return split_dict


def create_yahoo_imbalance(input_file, src_path, dest_path,input_classes,input_cols, 
                           new_distribution,incremental_pct,out_names):
    """
    create dataset with specific distribuiton
    @params
    file: file input
    src_path: path of the input file
    dest_path: output path for new dataset files
    input_cols: columns in input dataset
    distribution: percent to use in each class
    
    """
        
    df   = pd.read_csv(src_path+input_file,names =input_cols)
    data = df.groupby('class').count()
    class_size = data['questionlabel'].mean()
    y_train = df["class"].values
    
    #get ids for distribution
    index_dict = defaultdict(list)
    for i in range(len(y_train)):
        index_dict[y_train[i]].append(i)
    
   
    if os.path.exists(dest_path)==False:
        print ("-Adding folder:",dest_path)
        os.makedirs(dest_path)
        
    if input_classes!= len(data):
            print ("-Input dataset does not match with dataset")
            return
    
    for idx,percent in enumerate(incremental_pcts):
        split_dict = []
        split_dict = create_imbalance_index(new_distribution,class_size,percent,index_dict)
        #write dataset 
        imbalance_dataset = df.loc[split_dict,:] 
        out_name = out_names[idx]+input_file.split('.')[0]+".csv"
        print ("-Creating file at", dest_path+out_name)
        imbalance_dataset.to_csv(dest_path+out_name, sep=',',index=False,header=False)
        create_new_file_distribution(dest_path,input_classes,imbalance_dataset,out_names[idx] )
    
    print ("-Creation completed.")
    

def create_new_file_distribution(dest_path,num_classes,y_train, out_file):
    """
    compute the new distribution of the datset
    @param:
    dest_path: output path
    num_classes: number of classes in the dataset
    y_train: dataset input
    out_file: output filename
    """
    DEFAULT_SAMPLE_DIR ="/samples/"
    print('-Confirming new classes distribution')
    y_train= y_train["class"].values
    training_distr = dict(zip(range(num_classes),[0]*num_classes))
    print (training_distr)
    for i in range(y_train.shape[0]):
        c = y_train[i]-1
        training_distr[c] += 1
    print (training_distr)
    

    if os.path.exists(dest_path) == False:
        os.makedirs(dest_path)
        
    file = '{}/{}.distr'.format(dest_path,out_file)
    print('-Save new file classes distribution:', file)
    with open(file, 'wb') as handle:
        pickle.dump(training_distr, handle)


def split_test_to_val(src_path,dest_path, input_file, input_cols, test_size, out_file):
    """
    -Extract val dataset from test dataset using test_size input aprox 83% will be use as
    val.csv set
    @param:
    src_path: original yahoo dataset folder
    dest_path: output folder
    input_file: input file name
    input_cols: input column. names from input file
    test_size: target size in number of records
    out_file:  new file name    
    """
    NEW_TEST_PREFIX ="new_"
    df = pd.read_csv(src_path+input_file,names =input_cols)
    total = len(df)
    print ("-Generating validation from total test.csv:",total)
    # get random indices
    np.random.seed(SEED)
    indices = np.arange(0,total)
    np.random.shuffle(indices)

    set1 = indices[0:test_size]
    set2 = indices[test_size:total]

    dat1 = df.loc[set1,:] 
    dat2 = df.loc[set2,:]
    #write 
    dat1.to_csv(dest_path+input_file, sep=',',index=False,header=False)
    print ("-New file generated:",dest_path+input_file)
    dat2.to_csv(dest_path+out_file, sep=',',index=False,header=False)
    print ("-New file generated:",dest_path+out_file)



##############################
#Main driver
# perform yahoo dataset to make it imbalance 
# and incremental dataset.
##############################

if __name__ =="__main__":
    
    
    # 1.first extract val.csv dataet from test.csv
    input_file = 'test.csv'
    out_file   = 'val.csv'
    test_size  = 5000 #aprox 83% of test set only, 17% is used as val
    split_test_to_val(src_path,src_path, input_file, input_cols, test_size, out_file)
    
    # 2.transform - train
    input_file ='train.csv'
    create_yahoo_imbalance(input_file, src_path, dest_path,input_classes,input_cols, 
                           new_distribution,incremental_pcts,out_names)
    # 3.transform - train
    input_file ='val.csv'
    create_yahoo_imbalance(input_file, src_path, dest_path,input_classes,input_cols, 
                           new_distribution,incremental_pcts,out_names)
   
