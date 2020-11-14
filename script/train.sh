#!/bin/bash


: << '###'
*snapshot.py refer to DynSnap-cyc method in the paper
*snapshotA.py refer to DynSnap-cyc but the learning rate is updated per iteration
*snapshotB.py refer to DynSnap-step method in the paper
*dropout.py refer to MCDropout method in the paper

SingleBase: *snapshotB.py with --topK 1
ExtBagging: Run SingleBase M times with different --seed, then combine them
MCDropout: *dropout.py with mc_num = M
Snapshot: *snapshot.py without pruning
DynSnap-cyc: *snapshot.py with dynamic pruning
DynSnap-step: *snapshotB.py with dynamic pruning
###

# CIFAR10
workdir='/home/ubuntu/Projects/dynens' # Note: Change work directory based on need
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
workdir='/home/ubuntu/Projects/dynens' # Note: Change work directory based on need
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
workdir='/home/ubuntu/Projects/dynens' # Note: Change work directory based on need
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


