CC=gcc
CCFLAGS=-std=c99 -march=native -mtune=native -pg -Wall -Wextra -pedantic
CLFLAGS=


all: hmap.o qrand.o

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

#Removes all tildes ending files, objects codes and test executable
clean:
	rm -rf *~ *.o .*.swp gmon.out
