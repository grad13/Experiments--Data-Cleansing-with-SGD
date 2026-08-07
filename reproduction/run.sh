#!/bin/bash

<< COMMENTOUT
mkdir LOGs

nohup bash run.sh 0 0 mnist > LOGs/MNIST00.log &
nohup bash run.sh 1 1 mnist > LOGs/MNIST01.log &
nohup bash run.sh 2 2 mnist > LOGs/MNIST02.log &
nohup bash run.sh 3 3 mnist > LOGs/MNIST03.log &
nohup bash run.sh 4 4 mnist > LOGs/MNIST04.log &
nohup bash run.sh 5 5 mnist > LOGs/MNIST05.log &
nohup bash run.sh 6 6 mnist > LOGs/MNIST06.log &
nohup bash run.sh 7 7 mnist > LOGs/MNIST07.log &


nohup bash run.sh 0 0 cifar10 > LOGs/CIFAR00.log &
nohup bash run.sh 1 1 cifar10 > LOGs/CIFAR01.log &
nohup bash run.sh 2 2 cifar10 > LOGs/CIFAR02.log &
nohup bash run.sh 3 3 cifar10 > LOGs/CIFAR03.log &
nohup bash run.sh 4 4 cifar10 > LOGs/CIFAR04.log &
nohup bash run.sh 5 5 cifar10 > LOGs/CIFAR05.log &
nohup bash run.sh 6 6 cifar10 > LOGs/CIFAR06.log &
nohup bash run.sh 7 7 cifar10 > LOGs/CIFAR07.log &
COMMENTOUT

seed=$1
gpu=$2
data=$3
num_epochs=20
target_epoch=20
start_epoch=19

python -u code/train.py --gpu $gpu --output_path $data/0$gpu --dataset $data --seed $seed --num_epochs $num_epochs
python -u code/infl.py  --gpu $gpu --episode     $data/0$gpu --target_epoch $target_epoch
python -u code/eval.py  --gpu $gpu --episode     $data/0$gpu --start_epoch  $start_epoch  --target_epoch $num_epochs --end_epoch $num_epochs
