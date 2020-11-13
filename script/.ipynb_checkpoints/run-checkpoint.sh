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
datadirs=('cifar100_imbalance')
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
src='../../datasets/yahoo_answers_csv_with_val_imbalance/' # Note: Change work directory based on need
dat='/samples/80/'
dest_path='test_snap/80'
model='fasttext_yahoo_snapshot.py'
python $model --seed 10 --savedir ./$dest_path/run1/ --datafile $src$dat --gpu 0 --topk 10
python $model --seed 20 --savedir ./$dest_path/run2/ --datafile $src$dat --gpu 0 --topk 10
python $model --seed 30 --savedir ./$dest_path/run3/ --datafile $src$dat --gpu 0 --topk 10
python $model --seed 40 --savedir ./$dest_path/run4/ --datafile $src$dat --gpu 0 --topk 10
python $model --seed 50 --savedir ./$dest_path/run5/ --datafile $src$dat --gpu 0 --topk 10

dat='/samples/90/' # Note: Change work directory based on need
dest_path='out_imbalance_snap_exp1/90'
python $model --seed 40 --savedir ./$dest_path/run4/ --datafile $src$dat --gpu 5 --topk 10
python $model --seed 50 --savedir ./$dest_path/run5/ --datafile $src$dat --gpu 6 --topk 10
