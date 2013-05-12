CFLAGS=-march=native -mtune=native -Wall -Wextra -pedantic
CPPFLAGS=-Os

all: seq omp mpi 

seq: seq.cpp primeNumber.cpp common.h
	g++ $(CPPFLAGS) $^ -o seq -lm

omp: omp.c common.h
	gcc $(CPPFLAGS) -std=c99 -g -fopenmp $< -o omp -lm

mpi: mpi.cpp primeNumber.cpp common.h
	mpicxx $(CPPFLAGS) $^ -o mpi -lm

#Removes all tildes ending files, objects codes and test executable
clean:
	rm -rf *~ *.o .*.swp utest
