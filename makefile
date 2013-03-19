CC=gcc
CCFLAGS=-std=c99 -march=native -mtune=native -Os -Wall -Wextra -pedantic
CLD=-lm -lgmp -pthread


all: gauss_standart borwein_standart gauss_concurrent montecarlo_standart montecarlo_concurrent borwein_concurrent 

gauss_standart: gauss_standart.c
	$(CC) $(CCFLAGS) $< $(CLD) -o gauss_standart

gauss_concurrent: gauss_concurrent.c
	$(CC) $(CCFLAGS) $< $(CLD) -o gauss_concurrent

montecarlo_standart: montecarlo_standart.c
	$(CC) $(CCFLAGS) $< $(CLD) -o montecarlo_standart

montecarlo_concurrent: montecarlo_concurrent.c
	$(CC) $(CCFLAGS) $< $(CLD) -o montecarlo_concurrent

borwein_standart: borwein_standart.c
	$(CC) $(CCFLAGS) $< $(CLD) -o borwein_standart

borwein_concurrent: borwein_concurrent.c
	$(CC) $(CCFLAGS) $< $(CLD) -o borwein_concurrent


#Removes all tildes ending files, objects codes and test executable
clean:
	rm -rf *~ *.o .*.swp utest
