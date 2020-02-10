#!/bin/bash

# WORKDIR='/home/ubuntu/Projects/hybrid-ensemble'
WORKDIR='/scratch/lw8bn/hybrid_ensemble'

# python -W ignore compute_reproducibility.py -p1 '../output/cifar10_balance/DS1/simple-ensemble/run_1/prediction_0144.csv' -p2 '../output/cifar10_balance/DS1/simple-ensemble/run_2/prediction_0135.csv' -g '../output/cifar10_balance/DS1/simple-ensemble/run_1/target.csv'

# python -W ignore compute_accuracy.py -p '../output/cifar10_balance/DS1/simple-ensemble/run_1/prediction_0144.csv' -g '../output/cifar10_balance/DS1/simple-ensemble/run_1/target.csv' 

# # Compute Baseline accuracy
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
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
#         distr="/home/ubuntu/Projects/hybrid-ensemble/data/cifar100_imbalance/training_distr.pickle"
#         python -W ignore compute_accuracy.py -p $filename -g $gfile -o $ofile -d $distr
        
#         done
#     done
# done



## Compute Baseline reproducibility
# workdir=$WORKDIR/output
# datadirs=('yahoo_imbalance')
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
#     distr="$WORKDIR/data/$datadir/training_distr.pickle"
    
#     filename1=`find $workdir/$datadir/DS1/simple-ensemble/$run/ -name 'prediction*.csv'`
#     filename2=`find $workdir/$datadir/DS2/simple-ensemble/$run/ -name 'prediction*.csv'`
# #     python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $ofile -d $distr
#     python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -d $distr
    
#     filename1=`find $workdir/$datadir/DS1/simple-ensemble/$run/ -name 'prediction*.csv'`
#     filename2=`find $workdir/$datadir/DS3/simple-ensemble/$run/ -name 'prediction*.csv'`
# #     python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $ofile -d $distr
#     python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -d $distr
    
#     filename1=`find $workdir/$datadir/DS2/simple-ensemble/$run/ -name 'prediction*.csv'`
#     filename2=`find $workdir/$datadir/DS3/simple-ensemble/$run/ -name 'prediction*.csv'`
# #     python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $ofile -d $distr
#     python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -d $distr
        
#     done
# done

## Combine index files
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
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
# #             sed -i 's/cifar100_ResNet20v1_model-/prediction_/g;s/.h5/.csv/g' "$workdir/$datadir/$datafile/$modeldir/$run/index.csv"
#             prefix="$workdir/$datadir/$datafile/$modeldir/$run/"
# #             tail -5 "$workdir/$datadir/$datafile/$modeldir/$run/index.csv" | sed -e "s|^|"$prefix"|" >> "$workdir/$datadir/$datafile/$modeldir/index.csv"
#             done
# ##         rm "$workdir/$datadir/$datafile/$modeldir/index.csv"        
#         cat "$workdir/$datadir/$datafile/$modeldir/index.csv" >> "$workdir/$datadir/$datafile/super-ensemble/index.csv"
#         done
# ##     rm "$workdir/$datadir/$datafile/super-ensemble/index.csv"
#     done
# done

# ## Combination for ensemble methods
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
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
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
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
#                 distr="$WORKDIR/data/cifar100_imbalance/training_distr.pickle"
                
#                 python -W ignore compute_accuracy.py -p $pfile -g $gfile -o $ofile -d $distr
#                 done
#         done
#     done
# done

# ## Compute reproducibility for ensemble 
# workdir=$WORKDIR/output
# datadirs=('yahoo_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# # modeldirs=('simple-ensemble' 'snapshot' 'snapshot-A' 'snapshot-B' 'super-ensemble')
# modeldirs=('simple-ensemble' 'snapshot' 'snapshot-A' 'snapshot-B')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')

# # DS1-DS2
# for datadir in "${datadirs[@]}"
# do
#         echo $workdir $datadir
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS2/$modeldir/prediction_$method.csv"
# #                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -d $distr
#                done    
#         done
# done

# # DS1-DS3
# for datadir in "${datadirs[@]}"
# do
#         echo $workdir $datadir
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/$modeldir/prediction_$method.csv"
# #                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -d $distr
#                done    
#         done
# done

# ## DS2-DS3
# for datadir in "${datadirs[@]}"
# do
#         echo $workdir $datadir
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS2/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/$modeldir/prediction_$method.csv"
# #                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -d $distr
#                done    
#         done
# done

## Generate index files for sensitivity analysis on component numbers
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
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


# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
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
                
#                 dir="$workdir/$datadir/$datafile/sensitivity-ensemble-number/n160"
#                 python -W ignore combination.py -d $dir
   
#         done
# done

## Compute accuracy for ensemble 
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n5' 'n10' 'n20' 'n40' 'n160')
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
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"
                
#                 python -W ignore compute_accuracy.py -p $pfile -g $gfile -o $ofile -d $distr
#                 done
#         done
#     done
# done

# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n5' 'n10' 'n20' 'n40' 'n160')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS2/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done

# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n5' 'n10' 'n20' 'n40' 'n160')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done

# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n5' 'n10' 'n20' 'n40' 'n160')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS2/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/sensitivity-ensemble-number/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done


## =================================
## Sensitivity analysis on the number of trained learners in one shapshot learning (N)
# workdir=$WORKDIR/output
# prefix='/scratch/lw8bn/hybrid_ensemble/output/cifar100_imbalance/DS2/snapshot-A/run_5/'
# sed -e "s|^|"$prefix"|" < snapshot-A/run_5/index.csv > sensitivity-window-size/n20/index.csv
# prefix='/scratch/lw8bn/hybrid_ensemble/output/cifar100_imbalance/DS2/snapshot-A/run_8/'
# sed -e "s|^|"$prefix"|" < snapshot-A/run_8/index.csv > sensitivity-window-size/n30/index.csv
# prefix='/scratch/lw8bn/hybrid_ensemble/output/cifar100_imbalance/DS2/snapshot-A/run_9/'
# sed -e "s|^|"$prefix"|" < snapshot-A/run_9/index.csv > sensitivity-window-size/n40/index.csv

# prefix='/scratch/lw8bn/hybrid_ensemble/output/cifar100_imbalance/DS2/snapshot-A/run_5/'
# sed -e "s|^|"$prefix"|" < snapshot-A/run_5/index.csv > sensitivity-snapshot-number/n10/index.csv
# prefix='/scratch/lw8bn/hybrid_ensemble/output/cifar100_imbalance/DS2/snapshot-A/run_6/'
# sed -e "s|^|"$prefix"|" < snapshot-A/run_6/index.csv > sensitivity-snapshot-number/n20/index.csv
# prefix='/scratch/lw8bn/hybrid_ensemble/output/cifar100_imbalance/DS2/snapshot-A/run_7/'
# sed -e "s|^|"$prefix"|" < snapshot-A/run_7/index.csv > sensitivity-snapshot-number/n40/index.csv
# head -1 sensitivity-snapshot-number/n40/index.csv > sensitivity-snapshot-number/n1/index.csv
# head -5 sensitivity-snapshot-number/n40/index.csv > sensitivity-snapshot-number/n5/index.csv

## Combination
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# for datadir in "${datadirs[@]}"
# do
#         for datafile in "${datafiles[@]}"
#         do 
#                 echo "$workdir/$datadir/$datafile"
                
#                 dir="$workdir/$datadir/$datafile/sensitivity-window-size/n20"
#                 python -W ignore combination.py -d $dir
                
#                 dir="$workdir/$datadir/$datafile/sensitivity-window-size/n30"
#                 python -W ignore combination.py -d $dir
                
#                 dir="$workdir/$datadir/$datafile/sensitivity-window-size/n40"
#                 python -W ignore combination.py -d $dir
   
#         done
# done

# # Compute accuracy for ensemble 
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n20' 'n30' 'n40')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#     for datafile in "${datafiles[@]}"
#     do
#         echo $datafile
#         for modeldir in "${modeldirs[@]}"
#         do  
#                 for method in "${methods[@]}"
#                 do
#                 pfile="$workdir/$datadir/$datafile/sensitivity-window-size/$modeldir/prediction_$method.csv"
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 ofile="$workdir/$datadir/accuracy.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"
                
#                 python -W ignore compute_accuracy.py -p $pfile -g $gfile -o $ofile -d $distr
#                 done
#         done
#     done
# done

# echo 'DS1-DS2'
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n20' 'n30' 'n40')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/sensitivity-window-size/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS2/sensitivity-window-size/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done

# echo 'DS1-DS3'
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n20' 'n30' 'n40')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/sensitivity-window-size/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/sensitivity-window-size/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done

# echo 'DS2-DS2'
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n20' 'n30' 'n40')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS2/sensitivity-window-size/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/sensitivity-window-size/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done


## =======================================
## sensitivity-prune-factor
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n0' 'n1' 'n2' 'n3' 'n4' 'n5' 'n6' 'n7' 'n8' 'n9' 'n10')
# fractions=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# arraylength=${#fractions[@]}

# for datafile in "${datafiles[@]}"
# do
#         for ((i=0; i<${arraylength}; i++)); do
#             modeldir=${modeldirs[$i]}
#             fraction=${fractions[$i]}
#             inputfile="$WORKDIR/output/cifar100_imbalance/$datafile/snapshot-A/run_1/index.csv;$WORKDIR/output/cifar100_imbalance/$datafile/snapshot-A/run_2/index.csv;$WORKDIR/output/cifar100_imbalance/$datafile/snapshot-A/run_3/index.csv;$WORKDIR/output/cifar100_imbalance/$datafile/snapshot-A/run_4/index.csv;$WORKDIR/output/cifar100_imbalance/$datafile/snapshot-A/run_5/index.csv;$WORKDIR/output/cifar100_imbalance/$datafile/snapshot-A/run_6/index.csv;$WORKDIR/output/cifar100_imbalance/$datafile/snapshot-A/run_7/index.csv;$WORKDIR/output/cifar100_imbalance/$datafile/snapshot-A/run_8/index.csv;$WORKDIR/output/cifar100_imbalance/$datafile/snapshot-A/run_9/index.csv"
#             outputfile="$WORKDIR/output/cifar100_imbalance/$datafile/sensitivity-prune-factor/$modeldir/index.csv"
#             python prune.py -i $inputfile -o $outputfile -n 20 -a $fraction
#         done
# done

## Combination
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n0' 'n1' 'n2' 'n3' 'n4' 'n5' 'n6' 'n7' 'n8' 'n9' 'n10')
# for datadir in "${datadirs[@]}"
# do
#         for datafile in "${datafiles[@]}"
#         do 
#             for modeldir in "${modeldirs[@]}"
#             do
#                 echo "$workdir/$datadir/$datafile/$modeldir"
                
#                 dir="$workdir/$datadir/$datafile/sensitivity-prune-factor/$modeldir"
#                 python -W ignore combination.py -d $dir
                
#            done
#         done
# done

## Compute accuracy for ensemble 
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n0' 'n1' 'n2' 'n3' 'n4' 'n5' 'n6' 'n7' 'n8' 'n9' 'n10')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#     for datafile in "${datafiles[@]}"
#     do
#         echo $datafile
#         for modeldir in "${modeldirs[@]}"
#         do  
#                 for method in "${methods[@]}"
#                 do
#                 pfile="$workdir/$datadir/$datafile/sensitivity-prune-factor/$modeldir/prediction_$method.csv"
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 ofile="$workdir/$datadir/accuracy.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"
                
#                 python -W ignore compute_accuracy.py -p $pfile -g $gfile -o $ofile -d $distr
#                 done
#         done
#     done
# done

# echo 'DS1-DS2'
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n0' 'n1' 'n2' 'n3' 'n4' 'n5' 'n6' 'n7' 'n8' 'n9' 'n10')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/sensitivity-prune-factor/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS2/sensitivity-prune-factor/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done

# echo 'DS1-DS3'
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n0' 'n1' 'n2' 'n3' 'n4' 'n5' 'n6' 'n7' 'n8' 'n9' 'n10')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS1/sensitivity-prune-factor/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/sensitivity-prune-factor/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done

# echo 'DS2-DS2'
# workdir=$WORKDIR/output
# datadirs=('cifar100_imbalance')
# datafiles=('DS1' 'DS2' 'DS3')
# modeldirs=('n0' 'n1' 'n2' 'n3' 'n4' 'n5' 'n6' 'n7' 'n8' 'n9' 'n10')
# methods=('majority_voting' 'weighted_voting' 'averaging' 'weighted_averaging')
# for datadir in "${datadirs[@]}"
# do
#         for modeldir in "${modeldirs[@]}"
#         do 
#                 for method in "${methods[@]}"
#                 do
#                 gfile="$workdir/$datadir/DS1/simple-ensemble/run_1/target.csv"
#                 outfile="$workdir/$datadir/reproducibility.csv"
#                 distr="$WORKDIR/data/$datadir/training_distr.pickle"

#                 filename1="$workdir/$datadir/DS2/sensitivity-prune-factor/$modeldir/prediction_$method.csv"
#                 filename2="$workdir/$datadir/DS3/sensitivity-prune-factor/$modeldir/prediction_$method.csv"
#                 python -W ignore compute_reproducibility.py -p1 $filename1 -p2 $filename2 -g $gfile -o $outfile -d $distr

#                done    
#         done
# done