# This is the code for generating training, validating, testing data sets for hybrid-ensemble method.

from keras.datasets import cifar100
from collections import defaultdict
import pickle
import os
dirpath = os.getcwd()

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

train_size = int(x_train.shape[0]/100)
test_size = int(x_test.shape[0]/100)
idx = set(list(range(100)))
idx_less = idx-{10}

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

    
# Balancing training dataset
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

print(len(train_index_1), len(train_index_2), len(train_index_3))


datadir = '{}/../data/cifar100_balance'.format(dirpath)
x_train_new = x_train[train_index_1]
y_train_new = y_train[train_index_1]
DS_1 = ((x_train_new, y_train_new), (x_valid_new, y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS1'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_1, f)
    
x_train_new = x_train[train_index_2]
y_train_new = y_train[train_index_2]
DS_2 = ((x_train_new, y_train_new), (x_valid_new, y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS2'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_2, f)
    
x_train_new = x_train[train_index_3]
y_train_new = y_train[train_index_3]
DS_3 = ((x_train_new, y_train_new), (x_valid_new, y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS3'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_3, f)
    
# Imbalancing training dataset
# percents = [0.15834190084532787,
#  0.1176890564313457,
#  0.060386680374685864,
#  0.05670265021704365,
#  0.053275645419236926,
#  0.05023417866118346,
#  0.049691569568197404,
#  0.040153072880968706,
#  0.03449851496458762,
#  0.030000571167466305,
#  0.027216129769248348,
#  0.022261251999086133,
#  0.020676262280100527,
#  0.018720013708019193,
#  0.01837731322823852,
#  0.016178318482979213,
#  0.0150502627370345,
#  0.014978866803746861,
#  0.012608521818597215,
#  0.01208019191226868,
#  0.01178032899246059,
#  0.010523760566598128,
#  0.009938313913639479,
#  0.0090815627141878,
#  0.008624628741146905,
#  0.007582248115147362,
#  0.0071681517020790505,
#  0.006482750742517707,
#  0.005340415809915467,
#  0.005011994516792324,
#  0.004797806716929404,
#  0.004583618917066485,
#  0.004255197623943341,
#  0.004212360063970757,
#  0.004140964130683117,
#  0.003955334704135253,
#  0.003412725611149189,
#  0.0031271418779986295,
#  0.0030271875713959334,
#  0.002898674891478182,
#  0.002798720584875486,
#  0.002727324651587846,
#  0.0026273703449851502,
#  0.002527416038382454,
#  0.00227039067854695,
#  0.0021133196253141423,
#  0.0019990861320539188,
#  0.0019419693854238066,
#  0.0017706191455334707,
#  0.0016992232122458306,
#  0.0016278272789581908,
#  0.0015278729723554947,
#  0.001456477039067855,
#  0.001385081105780215,
#  0.001299405985835047,
#  0.0011851724925748231,
#  0.0011423349326022392,
#  0.0010281014393420153,
#  0.0009852638793694311,
#  0.0009281471327393193,
#  0.0008853095727667353,
#  0.0008567511994516793,
#  0.0008139136394790954,
#  0.0007996344528215675,
#  0.0007425177061914554,
#  0.0006996801462188715,
#  0.0006854009595613435,
#  0.0006425633995887596,
#  0.0005997258396161755,
#  0.0005426090929860636,
#  0.0005140507196710076,
#  0.0004997715330134797,
#  0.00048549234635595164,
#  0.0004569339730408956,
#  0.00044265478638336767,
#  0.00042837559972583967,
#  0.0004140964130683117,
#  0.00039981722641078374,
#  0.0003855380397532557,
#  0.00034270047978067177,
#  0.0003284212931231438,
#  0.00031414210646561573,
#  0.00029986291980808774,
#  0.0002713045464930318,
#  0.0002570253598355038,
#  0.00024274617317797582,
#  0.0002284669865204478,
#  0.0002284669865204478,
#  0.00021418779986291984,
#  0.00019990861320539187,
#  0.00018562942654786385,
#  0.00018562942654786385,
#  0.00018562942654786385,
#  0.00017135023989033589,
#  0.00017135023989033589,
#  0.00017135023989033589,
#  0.00015707105323280787,
#  0.00015707105323280787,
#  0.0001427918665752799,
#  0.0001427918665752799]

percents = [1,
 1,
 1,
 1,
 1,
 1,
 0.9,
 1,
 1,
 1,
 1,
 0.8,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 0.65,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 0.3,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 0.5,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 0.2,
 1,
 1,
 1,
 1,
 1,
 1,
 0.2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 0.1,
 0.1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1]

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

print(len(train_index_1), len(train_index_2), len(train_index_3))

datadir = '{}/../data/cifar100_imbalance'.format(dirpath)
x_train_new = x_train[train_index_1]
y_train_new = y_train[train_index_1]
DS_1 = ((x_train_new, y_train_new), (x_valid_new, y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS1'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_1, f)
    
x_train_new = x_train[train_index_2]
y_train_new = y_train[train_index_2]
DS_2 = ((x_train_new, y_train_new), (x_valid_new, y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS2'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_2, f)
    
x_train_new = x_train[train_index_3]
y_train_new = y_train[train_index_3]
DS_3 = ((x_train_new, y_train_new), (x_valid_new, y_valid_new), (x_test_new, y_test_new))
datafile = '{}/DS3'.format(datadir)
with open(datafile, 'wb') as f:
    pickle.dump(DS_3, f)