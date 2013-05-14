#include <stdio.h>
#include <stdlib.h>
#include "qrand.h"

unsigned long qrand_x = 123456789;
unsigned long qrand_y = 678912345;
unsigned long qrand_z = 987651234;

void qrand_seed(unsigned seed)
{
	qrand_x ^= seed;
	qrand_y ^= seed<<3;
	qrand_z ^= seed<<7;	
}

unsigned long qrand(void)
{
	unsigned long t;
	qrand_x ^= qrand_x<<16;
	qrand_x ^= qrand_x>>5;
	qrand_x ^= qrand_x<<1;
	t = qrand_x;
	qrand_x = qrand_y;
	qrand_y = qrand_z;
	qrand_z = t^qrand_x^qrand_y;
	return qrand_z;
}

void qrand_test(unsigned n, unsigned long long iterations)
{
	register unsigned long long i;
	unsigned *array;
	array = malloc(n*sizeof(unsigned));
	for (i=0; i<iterations; ++i)
		array[qrand()%n]+=1;
	for (i=0; i<n; i++)
	{
		printf("At position %llu: %u.\n", i, array[i]);
	}
	free(array);
}
