CC=g++
CCFLAGS=-Wall -pedantic -O3 
CLFLAGS=-lcuda
CUDAC=nvcc
CUDACFLAGS=

cpu: cpu_solver.o main.o smatrix.o
	$(CC) $(CCFLAGS) $^ -o cpu_solver

gpu: gpu_solver.o main.o smatrix.o
	$(CUDAC) $(CUDACFLAGS) $^ -o gpu_solver $(CLFLAGS)

main.o: main.cpp common.h
	$(CC) $(CCFLAGS) $< -c

smatrix.o: smatrix.cpp smatrix.h common.h
	$(CC) $(CCFLAGS) $< -c

cpu_solver.o: cpu_solver.cpp common.h
	$(CC) $(CCFLAGS) $< -c

gpu_solver.o: gpu_solver.cu common.h
	$(CUDAC) $(CUDACFLAGS) $< -c

#Removes all tildes ending files, objects codes and test executable
clean:
	rm -rf *~ *.o .*.swp cpu_solver gpu_solver 

