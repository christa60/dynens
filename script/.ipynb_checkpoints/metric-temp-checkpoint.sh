#!/bin/bash

WORKDIR='/home/ubuntu/Projects/hybrid-ensemble'
# WORKDIR='/scratch/lw8bn/hybrid_ensemble'

## =================================
## Sensitivity analysis on the number of trained learners in one shapshot learning (N)
## Generate index files for sensitivity analysis on N

## Compute dropout accuracy
workdir=$WORKDIR/output
datadirs=('cifar10_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
seeds=(22 34 46 58 60)
runs=('run_1' 'run_2' 'run_3')
for datadir in "${datadirs[@]}"
do
    for datafile in "${datafiles[@]}"
    do
        for i in "${!runs[@]}"
        do
        run="${runs[$i]}"
        seed="${seeds[$i]}"
        echo $workdir $datadir $datafile $run $seed
        filename=`find $workdir/$datadir/$datafile/dropout/$run/ -name 'prediction*.csv'`
        gfile="$workdir/$datadir/$datafile/dropout/$run/target.csv"
        ofile="$workdir/$datadir/accuracy.csv"
        distr="$WORKDIR/data/$datadir/training_distr.pickle"
        python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr
        
        done
    done
done

workdir=$WORKDIR/output
datadirs=('cifar10_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
method='dropout'
seeds=(22 34 46 58 60)
runs=('run_1' 'run_2' 'run_3')
for datadir in "${datadirs[@]}"
do
    for i in "${!runs[@]}"
    do
    run="${runs[$i]}"
    seed="${seeds[$i]}"
    echo $workdir $datadir $run $seed
    ofile="$workdir/$datadir/accuracy.csv"
    gfile="$workdir/$datadir/DS1/dropout/run_1/target.csv"
    #distr="$WORKDIR/data/$datadir/DS3.distr"
    distr="$WORKDIR/data/$datadir/training_distr.pickle"
    
    filename1=`find $workdir/$datadir/DS1/dropout/$run/ -name 'prediction*.csv'`
    filename2=`find $workdir/$datadir/DS2/dropout/$run/ -name 'prediction*.csv'`
    python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -d $distr
    
    filename1=`find $workdir/$datadir/DS1/dropout/$run/ -name 'prediction*.csv'`
    filename2=`find $workdir/$datadir/DS3/dropout/$run/ -name 'prediction*.csv'`
    python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -d $distr
    
    filename1=`find $workdir/$datadir/DS2/dropout/$run/ -name 'prediction*.csv'`
    filename2=`find $workdir/$datadir/DS3/dropout/$run/ -name 'prediction*.csv'`
    python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -d $distr
        
    done
done





# Compute dropout accuracy
# workdir=$WORKDIR/output
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# seeds=(22 34 46 58 60)
# runs=('run_1')
# for datadir in "${datadirs[@]}"
# do
#     for datafile in "${datafiles[@]}"
#     do
#         for i in "${!runs[@]}"
#         do
#         run="${runs[$i]}"
#         seed="${seeds[$i]}"
#         echo $workdir $datadir $datafile $run $seed
#         filename=`find $workdir/$datadir/$datafile/dropout/$run/ -name 'prediction*.csv'`
#         gfile="$workdir/$datadir/$datafile/dropout/$run/target.csv"
#         ofile="$workdir/$datadir/accuracy.csv"
#         distr="$WORKDIR/data/$datadir/training_distr.pickle"
#         python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr
        
#         done
#     done
# done

# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# seeds=(22 34 46 58 60)
# runs=('run_1')
# for datadir in "${datadirs[@]}"
# do
#     for datafile in "${datafiles[@]}"
#     do
#         for i in "${!runs[@]}"
#         do
#         run="${runs[$i]}"
#         seed="${seeds[$i]}"
#         echo $workdir $datadir $datafile $run $seed
#         filename=`find $workdir/$datadir/$datafile/dropout/$run/ -name 'prediction*.csv'`
#         gfile="$workdir/$datadir/$datafile/dropout/$run/target.csv"
#         ofile="$workdir/$datadir/accuracy.csv"
#         distr="$WORKDIR/data/$datadir/training_distr.pickle"
#         python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr
        
#         done
#     done
# done

# workdir=$WORKDIR/output
# datadir='yahoo_imbalance'
# run='run_1'
# datafile='DS1'
# filename=`find $workdir/$datadir/$datafile/dropout/$run/ -name 'prediction*.csv'`
# gfile="$workdir/$datadir/$datafile/dropout/$run/target.csv"
# ofile="$workdir/$datadir/accuracy.csv"
# distr="$WORKDIR/../output_Jan28/data/yahoo_answers_csv_with_val_imbalance/samples/80/training_distr.pickle"
# python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr

# datafile='DS2'
# filename=`find $workdir/$datadir/$datafile/dropout/$run/ -name 'prediction*.csv'`
# gfile="$workdir/$datadir/$datafile/dropout/$run/target.csv"
# ofile="$workdir/$datadir/accuracy.csv"
# distr="$WORKDIR/../output_Jan28/data/yahoo_answers_csv_with_val_imbalance/samples/90/training_distr.pickle"
# python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr

# datafile='DS3'
# filename=`find $workdir/$datadir/$datafile/dropout/$run/ -name 'prediction*.csv'`
# gfile="$workdir/$datadir/$datafile/dropout/$run/target.csv"
# ofile="$workdir/$datadir/accuracy.csv"
# distr="$WORKDIR/../output_Jan28/data/yahoo_answers_csv_with_val_imbalance/samples/100/training_distr.pickle"
# python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr

