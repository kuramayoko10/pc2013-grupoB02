#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <gmp.h>
#include "common.h"


int main(void)
{
	register unsigned i;
	mpf_t  x, y, p, aux1, aux2, sqrtx, invsqrtx;
	mpf_set_default_prec(BITS_PER_DIGIT*DIGITS);
	//mpf_set_default_prec(4096);
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
		if (i != 0)
		{
			/*y = (y*sqrt(x)+sqrt(1/x))/(y+1)*/
			mpf_mul(aux1, y, sqrtx);
			mpf_add(aux1, aux1, invsqrtx);
			mpf_add_ui(aux2, y, 1);
			mpf_div(y, aux1, aux2);		
		}
		/*x = 1/2(sqrt(x)+sqrt(1/x))*/
		mpf_add(x, sqrtx, invsqrtx);
		mpf_div_ui(x, x, 2);
		/*p = p*(x+1)/(y+1)*/
		mpf_add_ui(aux1, x, 1);
		mpf_add_ui(aux2, y, 1);
		mpf_div(aux1, aux1, aux2);
		mpf_mul(p, p, aux1);
	}
	mpf_out_str(stdout, 10, DIGITS, p);
	mpf_clear(x);
	mpf_clear(y);
	mpf_clear(p);
	mpf_clear(aux1);
	mpf_clear(aux2);
	mpf_clear(sqrtx);
	mpf_clear(invsqrtx);
	return 0;
}
