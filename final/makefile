CC=gcc
CCFLAGS=-std=c99 -march=native -mtune=native -Wall -Wextra -pedantic -O3 
CLFLAGS=
CUDAC=nvcc
CUDACFLAGS=

gpu: gpu_solver.o main.o smatrix.o
	$(CC) $(CCFLAGS) $^ -o cpu_solver $(CLFLAGS)

cpu: cpu_solver.o main.o smatrix.o
	$(CC) $(CCFLAGS) $^ -o cpu_solver $(CLFLAGS)

main.o: main.c common.h
	$(CC) $(CCFLAGS) $< -c

smatrix.o: smatrix.c smatrix.h common.h
	$(CC) $(CCFLAGS) $< -c

cpu_solver.o: cpu_solver.c common.h
	$(CC) $(CCFLAGS) $< -c

gpu_solver.o: gpu_solver.cu common.h
	$(CUDAC) $(CUDACFLAGS) $< -c

#Removes all tildes ending files, objects codes and test executable
clean:
	rm -rf *~ *.o .*.swp cpu_solver gpu_solver 
