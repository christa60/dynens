#!/bin/bash


#TODO:
# Check script comments/questions below
# Check seed settings for all experiments
# Check directory settings for all experiments
# Check run directory settings for all experiments
# Create scripts for Yahoo dataset
# Create script for dropout


# CIFAR10
workdir='/home/ubuntu/Projects/hybrid-ensemble' # Note: Change work directory based on need
datadirs=('cifar10_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
seeds=(22 34 46 58 60 70 80 90 100 110)
runs=('run_1' 'run_2' 'run_3' 'run_4' 'run_5' 'run_6' 'run_7' 'run_8' 'run_9' 'run_10')
for datadir in "${datadirs[@]}"
do
    for datafile in "${datafiles[@]}"
    do
        for i in "${!runs[@]}"
        do
        run="${runs[$i]}"
        seed="${seeds[$i]}"
        topk=10
        echo $workdir $datadir $datafile $run $seed
        python -W ignore cifar10_resnet-snapshot.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk
        python -W ignore cifar10_resnet-snapshotA.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot-A/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk
        python -W ignore cifar10_resnet-snapshotB.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot-B/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk

        done
    done
done

# CIFAR100
workdir='/home/ubuntu/Projects/hybrid-ensemble' # Note: Change work directory based on need
datadirs=('cifar100_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
seeds=(22 34 46 58 60 70 80 90 100 110)
runs=('run_1' 'run_2' 'run_3' 'run_4' 'run_5' 'run_6' 'run_7' 'run_8' 'run_9' 'run_10')
for datadir in "${datadirs[@]}"
do
    for datafile in "${datafiles[@]}"
    do
        for i in "${!runs[@]}"
        do
        run="${runs[$i]}"
        seed="${seeds[$i]}"
        topk=10
        echo $workdir $datadir $datafile $run $seed
        python -W ignore cifar100_resnet-snapshot.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk
        python -W ignore cifar100_resnet-snapshotA.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot-A/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk
        python -W ignore cifar100_resnet-snapshotB.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot-B/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk

        done
    done
done

# Yahoo
workdir='/home/ubuntu/Projects/hybrid-ensemble' # Note: Change work directory based on need
datadirs=('yahoo_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
seeds=(10 20 30 40 50 60 70 80 90 100)
runs=('run_1' 'run_2' 'run_3' 'run_4' 'run_5' 'run_6' 'run_7' 'run_8' 'run_9' 'run_10')
for datadir in "${datadirs[@]}"
do
    for datafile in "${datafiles[@]}"
    do
        for i in "${!runs[@]}"
        do
        run="${runs[$i]}"
        seed="${seeds[$i]}"
        topk=10
        echo $workdir $datadir $datafile $run $seed
        python -W ignore fasttext_yahoo_snapshot.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk
        python -W ignore fasttext_yahoo_snapshotA.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot-A/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk
        python -W ignore fasttext_yahoo_snapshotB.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot-B/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk

        done
    done
done


