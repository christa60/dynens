#!/bin/bash

# python -W ignore compute_reproducibility.py -p1 '../output/cifar10_balance/DS1/simple-ensemble/run_1/prediction_0144.csv' -p2 '../output/cifar10_balance/DS1/simple-ensemble/run_2/prediction_0135.csv' -g '../output/cifar10_balance/DS1/simple-ensemble/run_1/target.csv'

# python -W ignore compute_accuracy.py -p '../output/cifar10_balance/DS1/simple-ensemble/run_1/prediction_0144.csv' -g '../output/cifar10_balance/DS1/simple-ensemble/run_1/target.csv' 

## Compute Baseline accuracy
# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# seeds=(22 34 46 58 60)
# runs=('run_1' 'run_2' 'run_3' 'run_4' 'run_5')
# for datadir in "${datadirs[@]}"
# do
#     for datafile in "${datafiles[@]}"
#     do
#         for i in "${!runs[@]}"
#         do
#         run="${runs[$i]}"
#         seed="${seeds[$i]}"
#         echo $workdir $datadir $datafile $run $seed
#         filename=`find $workdir/$datadir/$datafile/simple-ensemble/$run/ -name 'prediction*.csv'`
#         gfile="$workdir/$datadir/$datafile/simple-ensemble/$run/target.csv"
#         ofile="$workdir/$datadir/accuracy.csv"
#         distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"
#         python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr
        
#         done
#     done
# done



# ## Compute Baseline reproducibility
# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# seeds=(22 34 46 58 60)
# runs=('run_1' 'run_2' 'run_3' 'run_4' 'run_5')
# for datadir in "${datadirs[@]}"
# do
#     for i in "${!runs[@]}"
#     do
#     run="${runs[$i]}"
#     seed="${seeds[$i]}"
#     echo $workdir $datadir $run $seed
#     ofile="$workdir/$datadir/accuracy.csv"
#     gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#     distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"
    
#     filename1=`find $workdir/$datadir/DS1/simple-ensemble/$run/ -name 'prediction*.csv'`
#     filename2=`find $workdir/$datadir/DS2/simple-ensemble/$run/ -name 'prediction*.csv'`
#     python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $ofile -d $distr
    
#     filename1=`find $workdir/$datadir/DS1/simple-ensemble/$run/ -name 'prediction*.csv'`
#     filename2=`find $workdir/$datadir/DS3/simple-ensemble/$run/ -name 'prediction*.csv'`
#     python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $ofile -d $distr
    
#     filename1=`find $workdir/$datadir/DS2/simple-ensemble/$run/ -name 'prediction*.csv'`
#     filename2=`find $workdir/$datadir/DS3/simple-ensemble/$run/ -name 'prediction*.csv'`
#     python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $ofile -d $distr
        
#     done
# done

# ## Combine index files
# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('simple-ensemble' 'snapshot' 'snapshot-A' 'snapshot-B')
# runs=('run_1' 'run_2' 'run_3' 'run_4' 'run_5')
# for datadir in "${datadirs[@]}"
# do
#     for datafile in "${datafiles[@]}"
#     do

#         for modeldir in "${modeldirs[@]}"
#         do
        
#             for run in "${runs[@]}"
#             do
#             sed -i 's/cifar10_ResNet20v1_model-/prediction_/g;s/.h5/.csv/g' "$workdir/$datadir/$datafile/$modeldir/$run/index.csv"
#             prefix="$workdir/$datadir/$datafile/$modeldir/$run/"
#             tail -5 "$workdir/$datadir/$datafile/$modeldir/$run/index.csv" | sed -e "s|^|"$prefix"|" >> "$workdir/$datadir/$datafile/$modeldir/index.csv"
#             done
# ##         rm "$workdir/$datadir/$datafile/$modeldir/index.csv"        
#         cat "$workdir/$datadir/$datafile/$modeldir/index.csv" >> "$workdir/$datadir/$datafile/super-ensemble/index.csv"
#         done
# ##     rm "$workdir/$datadir/$datafile/super-ensemble/index.csv"
#     done
# done

# ## Combination for ensemble methods
# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('simple-ensemble' 'snapshot' 'snapshot-A' 'snapshot-B' 'super-ensemble')
# for datadir in "${datadirs[@]}"
# do
#     for datafile in "${datafiles[@]}"
#     do
#         for modeldir in "${modeldirs[@]}"
#         do       
#             python -W ignore combination.py -d "$workdir/$datadir/$datafile/$modeldir"
#         done
#     done
# done

# ## Compute accuracy for ensemble 
# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('simple-ensemble' 'snapshot' 'snapshot-A' 'snapshot-B' 'super-ensemble')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#     for datafile in "${datafiles[@]}"
#     do
#         echo $workdir $datadir $datafile
#         for modeldir in "${modeldirs[@]}"
#         do  
#                 for method in "${methods[@]}"
#                 do   
#                 pfile="$workdir/$datadir/$datafile/$modeldir/prediction_$method.csv"
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 ofile="$workdir/$datadir/accuracy.csv"
#                 distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"
                
#                 python -W ignore compute_accuracy.py -p $pfile -g $gfile -o $ofile -d $distr
#                 done
#         done
#     done
# done

## Compute reproducibility for ensemble 
# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('simple-ensemble' 'snapshot' 'snapshot-A' 'snapshot-B' 'super-ensemble')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         echo $workdir $datadir
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS2/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr
#                done    
#         done
# done

# for datadir in "${datadirs[@]}"
# do
#         echo $workdir $datadir
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr
#                done    
#         done
# done

# for datadir in "${datadirs[@]}"
# do
#         echo $workdir $datadir
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS2/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr
#                done    
#         done
# done

## Generate index files for sensitivity analysis on component numbers
# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('sensitivity-ensemble-number')
# for datadir in "${datadirs[@]}"
# do
#         for datafile in "${datafiles[@]}"
#         do 
#                 indexfile="$workdir/$datadir/$datafile/super-ensemble/index.csv"
#                 seed=22
#                 oldworkdir='/home/ubuntu/Projects/hybrid-ensemble/output'
                
#                 number=5
#                 outfile="$workdir/$datadir/$datafile/sensitivity-ensemble-number/n5/index.csv"
#                 python -W ignore random_generate_index_by_number.py -i $indexfile -o $outfile -n $number -s $seed
#                 sed -i "s|$oldworkdir|$workdir|g" $outfile
                
#                 number=10
#                 outfile="$workdir/$datadir/$datafile/sensitivity-ensemble-number/n10/index.csv"
#                 python -W ignore random_generate_index_by_number.py -i $indexfile -o $outfile -n $number -s $seed
#                 sed -i "s|$oldworkdir|$workdir|g" $outfile
                
#                 number=20
#                 outfile="$workdir/$datadir/$datafile/sensitivity-ensemble-number/n20/index.csv"
#                 python -W ignore random_generate_index_by_number.py -i $indexfile -o $outfile -n $number -s $seed
#                 sed -i "s|$oldworkdir|$workdir|g" $outfile
                
#                 number=40
#                 outfile="$workdir/$datadir/$datafile/sensitivity-ensemble-number/n40/index.csv"
#                 python -W ignore random_generate_index_by_number.py -i $indexfile -o $outfile -n $number -s $seed
#                 sed -i "s|$oldworkdir|$workdir|g" $outfile
#         done
# done


# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# for datadir in "${datadirs[@]}"
# do
#         for datafile in "${datafiles[@]}"
#         do 
#                 echo "$workdir/$datadir/$datafile"
                
#                 dir="$workdir/$datadir/$datafile/sensitivity-ensemble-number/n5"
#                 python -W ignore combination.py -d $dir
                
#                 dir="$workdir/$datadir/$datafile/sensitivity-ensemble-number/n10"
#                 python -W ignore combination.py -d $dir
                
#                 dir="$workdir/$datadir/$datafile/sensitivity-ensemble-number/n20"
#                 python -W ignore combination.py -d $dir
                
#                 dir="$workdir/$datadir/$datafile/sensitivity-ensemble-number/n40"
#                 python -W ignore combination.py -d $dir
   
#         done
# done

# ## Compute accuracy for ensemble 
# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n5' 'n10' 'n20' 'n40')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#     for datafile in "${datafiles[@]}"
#     do
#         for modeldir in "${modeldirs[@]}"
#         do  
#                 for method in "${methods[@]}"
#                 do
#                 pfile="$workdir/$datadir/$datafile/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 ofile="$workdir/$datadir/accuracy.csv"
#                 distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"
                
#                 python -W ignore compute_accuracy.py -p $pfile -g $gfile -o $ofile -d $distr
#                 done
#         done
#     done
# done

# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n5' 'n10' 'n20' 'n40')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS2/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done

# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n5' 'n10' 'n20' 'n40')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done

# workdir='/home/ubuntu/Projects/hybrid-ensemble/output'
# datadirs=('cifar10_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n5' 'n10' 'n20' 'n40')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar10_imbalance/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS2/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done