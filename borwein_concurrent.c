#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <pthread.h>
#include <time.h>
#include "common.h"

pthread_t t1, t2;
unsigned i;
mpf_t  x, y, p, aux1, aux2, sqrtx, invsqrtx;
clock_t end, begin, b1, b2, e1, e2;

void *thread1(void *param);
void *thread2(void *param);

int main(void)
{
	clock_t begin, end;
	mpf_set_default_prec(BITS_PER_DIGIT*DIGITS);
	//mpf_set_default_prec(4096);
	begin = clock();
	mpf_init(x);
	mpf_init(y);
	mpf_init(p);
	mpf_init(aux1);
	mpf_init(aux2);
	mpf_init(sqrtx);
	mpf_init(invsqrtx);
	/* x = sqrt(2)*/
	mpf_set_ui(x, 2);
	mpf_sqrt(x, x);
	/* y = sqrt(sqrt(2)) = sqrt(x)*/
	mpf_sqrt(y, x);
	/* p = 2 + sqrt(2) = 2 + x*/
	mpf_add_ui(p, x, 2);
	for (i=0; i<24; i++)
	{
		mpf_sqrt(sqrtx, x);
		mpf_ui_div(invsqrtx, 1, sqrtx);
		pthread_create(&t1, NULL, thread1, NULL);
		pthread_create(&t2, NULL, thread2, NULL);
		pthread_join(t1, NULL);
		pthread_join(t2, NULL);
		mpf_div(p, aux1, aux2);
	}
	mpf_out_str(stdout, 10, DIGITS, p);
	mpf_clear(x);
	mpf_clear(y);
	mpf_clear(p);
	mpf_clear(aux1);
	mpf_clear(aux2);
	mpf_clear(sqrtx);
	mpf_clear(invsqrtx);
	end = clock();
	printf("Took %lfs\n", (double)(end-begin)/CLOCKS_PER_SEC);
	pthread_exit(0);
}


void *thread1(void *param)
{
	if (i != 0)
	{
		mpf_mul(aux2, y, sqrtx);
		mpf_add(aux2, aux2, invsqrtx);
		mpf_add_ui(y, y, 1);
		mpf_div(y, aux2, y);		
	}
	mpf_add_ui(aux2, y, 1);
	pthread_exit(0);
}

void *thread2(void *param)
{
	mpf_add(x, sqrtx, invsqrtx);
	mpf_div_ui(x, x, 2);
	mpf_add_ui(aux1, x, 1);
	mpf_mul(aux1 ,p, aux1);
	pthread_exit(0);
}

