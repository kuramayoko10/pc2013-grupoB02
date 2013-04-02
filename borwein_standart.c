#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <time.h>
#include "common.h"

unsigned i;
mpf_t  x, y, p, aux1, aux2, sqrtx, invsqrtx;
clock_t end, begin;

void thread1(void);
void thread2(void);

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
		mpf_add(x, sqrtx, invsqrtx);
		mpf_ui_div(invsqrtx, 1, sqrtx);
		thread1();
		thread2();
		mpf_div(p, aux1, aux2);
        
        //Para ver os valores de pi a cada iteracao
        //mpf_out_str(stdout, 10, DIGITS, p);
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
	printf("Took %lf.\n", (double)(end-begin)/CLOCKS_PER_SEC);
	return 0;
}


void thread1(void)
{
	if (i != 0)
	{
		mpf_mul(aux2, y, sqrtx);
		mpf_add(aux2, aux2, invsqrtx);
		mpf_add_ui(y, y, 1);
		mpf_div(y, aux2, y);		
	}
	mpf_add_ui(aux2, y, 1);
}

void thread2(void)
{
	mpf_div_ui(x, x, 2);
	mpf_add_ui(aux1, x, 1);
	mpf_mul(aux1 ,p, aux1);
}
