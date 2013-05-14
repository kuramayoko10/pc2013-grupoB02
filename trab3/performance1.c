#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "qrand.h"
#include "common.h"


int main(void)
{
	clock_t begin, end;
	unsigned long long i;
	qrand_seed(time(NULL));
	qrand_test(2, 1000000);
	return 0;
}
