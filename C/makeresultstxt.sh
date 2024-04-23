#!/bin/bash

#Runs 64-2048 mat mult for cuda
rm results.txt
./matmult.sh 64
sleep 1
./matmult.sh 128
sleep 2
./matmult.sh 256
sleep 3
./matmult.sh 512
sleep 5
./matmult.sh 1024
sleep 10
./matmult.sh 2048
sleep 10
./matmult.sh 4096

cat results.txt

exit 0
