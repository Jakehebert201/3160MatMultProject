#!/bin/bash 


#if args != 1
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 MATRIX_SIZE"
  exit 1
fi 

#Sets MATRIX_SIZE to first arg 
MATRIX_SIZE="$1"


#new line 
echo "" >> results.txt

#Runs with args
#I want to only show MatrixA and Performance= lines
./matrixMul -wA=$MATRIX_SIZE -hA=$MATRIX_SIZE -wB=$MATRIX_SIZE -hB=$MATRIX_SIZE | grep -E "MatrixA|Performance"   >> results.txt

