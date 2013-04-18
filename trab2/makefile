CC=clang
CCFLAGS=-std=c99 -march=native -mtune=native -Os -Wall -Wextra -pedantic
CLFLAGS=

all: sstack.o squeue.o llist.o hmap.o

sstack.o: sstack.c sstack.h common.h
	$(CC) $(CCFLAGS) $< -c

squeue.o: squeue.c squeue.h common.h
	$(CC) $(CCFLAGS) $< -c

llist.o: llist.c llist.h common.h
	$(CC) $(CCFLAGS) $< -c

hmap.o: hmap.c hmap.h common.h
	$(CC) $(CCFLAGS) $< -c

utest: utest.c all
	$(CC) $(CCFLAGS) $(CLFLAGS) -pg $< hmap.o -o utest

#Removes all tildes ending files, objects codes and test executable
clean:
	rm -rf *~ *.o .*.swp utest
