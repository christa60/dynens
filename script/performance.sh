#!/bin/bash

WORKDIR='/home/ubuntu/Projects/dynens' # Note: Change work directory based on need


# Compute accuracy (ACC) for ensemble 
workdir=$WORKDIR/output
datadirs=('cifar100_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
methodname='dynens-cyc'
modeldirs=('')
methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
for datadir in "${datadirs[@]}"
do
    for datafile in "${datafiles[@]}"
    do
        echo $datadir/$datafile
        for modeldir in "${modeldirs[@]}"
        do  
                for method in "${methods[@]}"
                do
                pfile="$workdir/$datadir/$datafile/$methodname/prediction_$method.csv"
                gfile="$workdir/$datadir/$datafile/$methodname/run_1/target.csv"
                ofile="$workdir/$datadir/accuracy.csv"
                
                python -W ignore compute_acc.py -p $pfile -g $gfile -o $ofile
                done
        done
    done
done


# Compute consistency (CON) and correct-consistency (ACC-CON) for ensemble
echo 'DS1-DS2'
workdir=$WORKDIR/output
datadirs=('cifar100_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
modeldirs=('dynens-cyc')
methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
for datadir in "${datadirs[@]}"
do
        for modeldir in "${modeldirs[@]}"
        do 
                for method in "${methods[@]}"
                do
                gfile="$workdir/$datadir/$datafile/$methodname/run_1/target.csv"
                outfile="$workdir/$datadir/consistency.csv"

                filename1="$workdir/$datadir/DS1/$modeldir/prediction_$method.csv"
                filename2="$workdir/$datadir/DS2/$modeldir/prediction_$method.csv"
                python -W ignore compute_con.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile


               done    
        done
done

echo 'DS1-DS3'
workdir=$WORKDIR/output
datadirs=('cifar100_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
modeldirs=('dynens-cyc')
methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
for datadir in "${datadirs[@]}"
do
        for modeldir in "${modeldirs[@]}"
        do 
                for method in "${methods[@]}"
                do
                gfile="$workdir/$datadir/$datafile/$methodname/run_1/target.csv"
                outfile="$workdir/$datadir/consistency.csv"

                filename1="$workdir/$datadir/DS1/$modeldir/prediction_$method.csv"
                filename2="$workdir/$datadir/DS3/$modeldir/prediction_$method.csv"
                python -W ignore compute_con.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile

               done    
        done
done

echo 'DS2-DS3'
workdir=$WORKDIR/output
datadirs=('cifar100_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
modeldirs=('dynens-cyc')
methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
for datadir in "${datadirs[@]}"
do
        for modeldir in "${modeldirs[@]}"
        do 
                for method in "${methods[@]}"
                do
                gfile="$workdir/$datadir/$datafile/$methodname/run_1/target.csv"
                outfile="$workdir/$datadir/consistency.csv"

                filename1="$workdir/$datadir/DS2/$modeldir/prediction_$method.csv"
                filename2="$workdir/$datadir/DS3/$modeldir/prediction_$method.csv"
                python -W ignore compute_con.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile

               done    
        done
done
