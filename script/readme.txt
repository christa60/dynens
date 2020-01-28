This is for hybrid ensemble experiments.

## Activate conda env for the codes
source activate tensorflow_p36

## Generate training dataset (imbalancing datasets)
generate_training_dataset_cifar10.py
generate_training_dataset_cifar100.py

## Generate class distribution for imbalance dataset. This will be used when computing majority and minority metrics. 
generate_class_distribution.py

## Run model training and predicting. Run different models on different datasets. Please modify the commands in run.sh.
./run.sh

## Outputs combination, accuracy and reproducibility computation. Please do it step by step.
metric.sh

## Choose a number of components K, then randomly select K single learners from the pool of single learners.
python -W ignore random_generate_index_by_number.py -i $indexfile -o $outfile -n $number -s $seed

## Compute accuracy
python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr

## Compute reproducibility
python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr
