#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "qrand.h"

int main(void)
{
	clock_t begin, end;
	unsigned long long i;
	begin = clock();
	for (i=0; i<1000000; i++)
	{
		qrand();
	}
	end = clock();
	printf("Took %fs.\n", (float)(end-begin)/CLOCKS_PER_SEC);
	begin = clock();
	for (i=0; i<100000000; i++)
	{
		rand();
	}
	end = clock();
	printf("Took %fs.\n", (float)(end-begin)/CLOCKS_PER_SEC);

	return 0;
}
