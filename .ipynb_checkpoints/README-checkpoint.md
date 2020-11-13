# Dynamic snapshot ensemble

Implementation of the paper [Wisdom of the Ensemble: Improving Consistency of Deep Learning Models](https://papers.nips.cc/paper/2020/hash/e464656edca5e58850f8cec98cbb979b-Abstract.html)

# Description

This paper studies consistency and correct-consistency of a learning model. We prove that consistency and correct-consistency of an ensemble learner is not less than the average consistency and correct-consistency of individual learners and correct-consistency can be improved with a probability by combining learners with accuracy not less than the average accuracy of ensemble component learners. To validate the theory using three datasets and two state-of-the-art deep learning classifiers we propose an efficient dynamic snapshot ensemble method and demonstrate its value.

The theoretical findings please refer to our paper ([here](https://papers.nips.cc/paper/2020/hash/e464656edca5e58850f8cec98cbb979b-Abstract.html)).
In this repository, we focus on describing the proposed dynamic snapshot ensemble method. 

Preliminaries:
Snapshot ensemble learning [huang2017snapshot](https://openreview.net/pdf?id=BJYwwY9ll): instead of training N neural networks independently, the optimizer converges N times to local optima along its optimization path, thereby training multiple single learners at no additional cost.

Extended bagging: training on entire dataset with random shuffling and with random initialization of the neural network parameters.

Our method:
We extend the snapshot learning in two dimensions - learning rate schedule (cyclic annealing schedule and step-wise decay schedule) and snapshot saving strategy (cyclic snapshot and top-N snapshot).
We implement two snapshot learning methods: *dynens-cyc* uses cyclic annealing schedule and cyclic snapshot strategy with $t$ as epoch number; *dynens-step* uses step-wise decay schedule and top-N snapshot strategy.

**Pruning criteria**
A single learner is included in the final ensemble learner if its validation accuracy is larger than the pruning threshold defined as:<br>
<img src='prune.png'>

**Dynamic snapshot ensemble**
Comining extended bagging technique and snapshot learning technique with dynamic pruning. The proposed ensemble method can achieve better accuracy, consistency, correct-consistency without compromising computational cost.<br>
<img src='framework.png'>

# Performance
Single learner: SingleBase 
Ensemble learner: ExtBagging, MCDropout, Snapshot, DynSnap-cyc (ours), DynSnap-step
(ours) 
Accuracy (ACC); Consistency (CON); Correct-Consistency (ACC-CON) 
Combination methods: Majority Voting (MV); Weighted Majority Voting (WMV); Averaging (AVG); Weighted Averaging (WAVG)

<img src='result.png' width=90%>


# Usage

This paper conducts experiments using three datasets (CIFAR10, CIFAR100, and YAHOO!Answers) and two state-of-the-art models (ResNet and fastText). We use ResNet for CIFAR10 and CIFAR100 and fastText for for YAHOO!Answers. To simulate online data streams with imbalanced class distribution, we reorganize the datasets so that three class imbalanced training sets are generated for each dataset.

## Data preparation

```
python generate_training_dataset_cifar10.py
python generate_training_dataset_cifar100.py
```

## Train and predict

Please refer to *train.sh* and make corresponding changes as needed. The seed values and hyperparameter settings to reproduce the experiments in the paper are listed in the codes. 

```
train.sh
```

## Post-prune 

To experiment on \beta, we separate the pruning step with the single learning generation step. In practical application, these two steps can be integrate into one step.

```
ensemble.sh
```

## Evaluate model performance

```
performance.sh
```

## Discrepancies

```
generage_class_distribution.py
compute_reproducibility.py
random_generate_index_by_number.py
```

# Requirements
python 3.6.5
numpy 1.15.4
pandas 0.24.2
tensorflow 1.13.1 
keras 2.2.4
scikit-learn 0.19.1
scipy 1.1.0
