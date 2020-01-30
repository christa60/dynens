#!/bin/bash
# Scrip to run experiments 
# with different datasets, topk and gpu
# our servers have 8 GPUs

# author tere
# parameters
src='../../datasets/yahoo_answers_csv_with_val_imbalance/'
dat='/samples/80/'
dest_path='test_snap/80'
model='fasttext_yahoo_snapshot.py'

#python $model --seed 10 --savedir ./$dest_path/run1/ --datafile $src$dat --gpu 1 --topk 2

nohup python $model --seed 10 --savedir ./$dest_path/run1/ --datafile $src$dat --gpu 0 --topk 10 >/dev/null 2>&1 &
sleep 5
nohup python $model --seed 20 --savedir ./$dest_path/run2/ --datafile $src$dat --gpu 1 --topk 10 >/dev/null 2>&1 &
sleep 5
nohup python $model --seed 30 --savedir ./$dest_path/run3/ --datafile $src$dat --gpu 2 --topk 10 >/dev/null 2>&1 &
sleep 5 
nohup python $model --seed 40 --savedir ./$dest_path/run4/ --datafile $src$dat --gpu 3 --topk 10 >/dev/null 2>&1 &
sleep 5
nohup python $model --seed 50 --savedir ./$dest_path/run5/ --datafile $src$dat --gpu 4 --topk 10 >/dev/null 2>&1 &
sleep 10

dat='/samples/90/'
dest_path='out_imbalance_snap_exp1/90'
nohup python $model --seed 40 --savedir ./$dest_path/run4/ --datafile $src$dat --gpu 5 --topk 10 >/dev/null 2>&1 &
sleep 5
nohup python $model --seed 50 --savedir ./$dest_path/run5/ --datafile $src$dat --gpu 6 --topk 10 >/dev/null 2>&1 &

