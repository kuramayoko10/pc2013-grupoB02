CFLAGS=-march=native -mtune=native -Wall -Wextra -pedantic

all: seq omp mpi 

seq: seq.c common.h
	gcc $(CFLAGS) -std=c99 $< -o seq -lm

omp: omp.c common.h
	gcc $(CFLAGS) -std=c99 -g -fopenmp $< -o omp -lm

mpi: mpi.cpp common.h
	mpig++ $(CFLAGS) $< -o mpi -lm

#Removes all tildes ending files, objects codes and test executable
clean:
	rm -rf *~ *.o .*.swp utest
