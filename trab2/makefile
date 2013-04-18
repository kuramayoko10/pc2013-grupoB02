CFLAGS=-march=native -mtune=native -Os -Wall -Wextra -pedantic

all: seq omp mpi 

seq: seq.cpp common.h
	g++ $(CFLAGS) $< -o seq

omp: omp.cpp common.h
	g++ $(CFLAGS) -fopenmp $< -o omp

mpi: mpi.cpp common.h
	mpig++ $(CFLAGS) $< -o mpi

utest: utest.c all
	$(CC) $(CCFLAGS) $(CLFLAGS) -pg $< hmap.o -o utest

#Removes all tildes ending files, objects codes and test executable
clean:
	rm -rf *~ *.o .*.swp utest
