CC=gcc
CCFLAGS=-std=c99 -march=native -mtune=native -Wall -Wextra -pedantic -Os -pg
CLFLAGS=


all: hmap.o qrand.o process.o
	$(CC) $(CCFLAGS) $^ -o process $(CLFLAGS)

all1: qrand.o
	cp performance1.c performance.c
	$(CC) $(CCFLAGS) $^ performance.c -o performance 
	rm performance.c

all2: qrand.o
	cp performance2.c performance.c
	$(CC) $(CCFLAGS) $^ performance.c -o performance 
	rm performance.c

hmap.o: hmap.c hmap.h common.h
	$(CC) $(CCFLAGS) $< -c

qrand.o: qrand.c qrand.h common.h
	$(CC) $(CCFLAGS) $< -c

process.o: process.c process.h common.h
	$(CC) $(CCFLAGS) $< -c

process: hmap process.c process.h common.h
	$(CC) $(CCFLAGS) $< -o process

#Removes all tildes ending files, objects codes and test executable
clean:
	rm -rf *~ *.o .*.swp gmon.out process
