# Dynamic snapshot ensemble

## Activate conda env for the codes
(AWS p3.2Xlarge) source activate tensorflow_p36
OR
tensorflow 1.13.1
keras 2.2.4

## Generate balanced and imbalanced training dataset. Generate validation and testing datasets
- python generate_training_dataset_cifar10.py
- python generate_training_dataset_cifar100.py

## Generate class distribution for imbalance dataset. 
### To be used while computing majority and minority class metrics. 
- python generate_class_distribution.py

## Run model training and inference. 
### Run different models on different datasets. Please modify the commands in run.sh.
#### Note: *-snapshot.py corresponds to dynens-cyc learning with t as iteration number
#### Note: *-snapshotA.py corresponds to dynens-cyc
#### Note: *-snapshotB.py corresponds to dynens-step. 
./run.sh

## Pruning algorithm. To experiment on \beta, we separate the pruning step with the single learning generation step. In practical application, these two steps can be integrate into one step.
python prune-avg.py -i $inputfile -n 20 -o $outputfile

## Outputs combination, accuracy and reproducibility computation. Please do it step by step.
metric.sh

## Choose a number of components K, then randomly select K single learners from the pool of single learners. (discrepancy)
python -W ignore random_generate_index_by_number.py -i $indexfile -o $outfile -n $number -s $seed

## Compute accuracy
python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr

## Compute reproducibility
python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr
