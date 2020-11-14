#!/bin/bash

WORKDIR='/home/ubuntu/Projects/dynens' # Note: Change work directory based on need

# Dynamic pruning
workdir=$WORKDIR/output
datafiles=('DS1' 'DS2' 'DS3')
for datafile in "${datafiles[@]}"
do
        echo $datafile
        datadir='cifar100_imbalance'
        method='dynens-cyc'
        inputfile="$WORKDIR/output/$datadir/$datafile/$method/run_1/index.csv;$WORKDIR/output/$datadir/$datafile/$method/run_2/index.csv;$WORKDIR/output/$datadir/$datafile/$method/run_3/index.csv;$WORKDIR/output/$datadir/$datafile/$method/run_4/index.csv;$WORKDIR/output/$datadir/$datafile/$method/run_5/index.csv;$WORKDIR/output/$datadir/$datafile/$method/run_6/index.csv;$WORKDIR/output/$datadir/$datafile/$method/run_7/index.csv;$WORKDIR/output/$datadir/$datafile/$method/run_8/index.csv;$WORKDIR/output/$datadir/$datafile/$method/run_9/index.csv;$WORKDIR/output/$datadir/$datafile/$method/run_10/index.csv"
        
        outputfile="$WORKDIR/output/$datadir/$datafile/$method/index.csv"
        
        python prune-avg.py -i $inputfile -n 20 -o $outputfile
done

# Top-N pruning
workdir=$WORKDIR/output
datafiles=('DS1' 'DS2' 'DS3')
for datafile in "${datafiles[@]}"
do
        datadir='cifar100_imbalance'
        method='dynens-cyc'
        echo $datadir $datafile $method
        inputfile="$WORKDIR/output/$datadir/$datafile/$method/run_1/index.csv"
        outputfile="$WORKDIR/output/$datadir/$datafile/$method/index.csv"
        python prune.py -i $inputfile -n 20 -o $outputfile
done


# Combination to generate ensemble outputs
workdir=$WORKDIR/output
datadirs=('cifar100_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
method='dynens-cyc'
for datadir in "${datadirs[@]}"
do
        for datafile in "${datafiles[@]}"
        do 
                echo "$workdir/$datadir/$datafile/$method"
                
                dir="$workdir/$datadir/$datafile/$method"
                python -W ignore combination.py -d $dir
   
        done
done