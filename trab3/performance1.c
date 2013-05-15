#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "qrand.h"

#define ITERATIONS 100000


int main(void)
{
	clock_t begin, end;
	unsigned long long i;
	char string[6];
	begin = clock();
	qrand_seed(time(NULL));
	for (i=0; i<ITERATIONS; i++)
	{
		qrand_word(string, 6);
	}
	end = clock();
	printf("Took %fs.\n", (float)(end-begin)/CLOCKS_PER_SEC);
	return 0;
}
