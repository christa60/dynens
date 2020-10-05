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
datadirs=('cifar10_balance' 'cifar10_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
seeds=(22 34 46 58 60)
runs=('run_1' 'run_2' 'run_3' 'run_4' 'run_5')
for datadir in "${datadirs[@]}"
do
    for datafile in "${datafiles[@]}"
    do
        for i in "${!runs[@]}"
        do
        run="${runs[$i]}"
        seed="${seeds[$i]}"
        topk=5
        echo $workdir $datadir $datafile $run $seed
        python -W ignore cifar10_resnet-snapshot.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk
        python -W ignore cifar10_resnet-snapshotA.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot-A/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk
        python -W ignore cifar10_resnet-snapshotB.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/snapshot-B/$run" --datafile "$workdir/data/$datadir/$datafile" --topk $topk

        # TODO: Do we need this?
        # python -W ignore cifar10_resnet-snapshotB.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/simple-ensemble/$run" --datafile "$workdir/data/$datadir/$datafile" --topk 1
        done
    done
done

# CIFAR100
workdir='/home/ubuntu/Projects/hybrid-ensemble' # Note: Change work directory based on need
datadirs=('cifar100_balance' 'cifar100_imbalance')
datafiles=('DS1' 'DS2' 'DS3')
seeds=(22 34 46 58 60)
runs=('run_1' 'run_2' 'run_3' 'run_4' 'run_5')
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

        # TODO: Do we need this?
        # python -W ignore cifar100_resnet-snapshotB.py --seed $seed --savedir "$workdir/output/$datadir/$datafile/simple-ensemble/$run" --datafile "$workdir/data/$datadir/$datafile" --topk 1
        done
    done
done

# Yahoo
# TODO: Need to write scipts
