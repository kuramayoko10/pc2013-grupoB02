#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qrand.h"


void qrand_seed(unsigned seed)
{
	qrand_x ^= seed;
	qrand_y ^= seed<<3;
	qrand_z ^= seed<<7;	
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


void qrand_word(char *s)
{
	unsigned i, lenght;
	lenght = rand()%TOTAL_FIVE;
	if (lenght > TOTAL_FOUR)
		lenght = 5;	
	else if (lenght > TOTAL_THREE)
		lenght = 4;
	else if (lenght > TOTAL_TWO)
		lenght = 3;
	else if (lenght > TOTAL_ONE)
		lenght = 2;
	else 
		lenght = 1;

	memset(s+lenght, '\0', 30-lenght);
	for (i=0; i<lenght; ++i)
	{
		s[i]='a'+(qrand())%26;	
	}
	return;
}
