#!/bin/bash
make clean
make
for i in $(seq 1 10)
do
	./cpu_solver 1000 >> $1 
done
