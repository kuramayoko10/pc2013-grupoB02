#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <gmp.h>

int main(void)
{
	register unsigned i;
	mpf_t  a, s, r, aux1, aux2;
	mpf_init(a);
	mpf_init(r);
	mpf_init(s);
	mpf_init(aux1);
	mpf_init(aux2);
	/* a = 1.0/3.0*/
	mpf_set_ui(a, 1);
	mpf_div_ui(a, a, 3);
	/* s = (sqrt(3)-1)/2.0*/
	mpf_sqrt_ui(aux1, 3);
	mpf_sub_ui(aux1, aux1, 1);
	mpf_div_ui(s, aux1, 2);
	for (i=0; i<15; i++)
	{
		/*r = 3.0/(1.0+2.0*powl(1.0-s*s*s,1.0/3.0))*/
		mpf_set_ui(aux2, 1);
		mpf_div_ui(aux2, aux2, 3);
		mpf_pow_ui(aux1, s, 3);
		mpf_ui_sub(aux1, 1, aux1);
		mpf_pow(aux1, aux1, aux2);
		mpf_mul_ui(aux1, aux1, 2);
		mpf_add_ui(aux1, aux1, 1);
		mpf_ui_div(r, 3, aux1);
		/*s = (r-1.0)/2.0*/
		mpf_sub_ui(aux1, r, 1);
		mpf_div_ui(s, aux1, 2);
		/*a = r*r*a-powl(3.0, i)*(r*r-1.0)*/
		mpf_pow_ui(aux1, r, 2);
		mpf_mult(aux1, aux1, a);
		mpf_pow_ui(aux2, r, 2);
		mpf_sub_ui(aux2, aux2, 1);
		mpf_mult(aux2, powl(3.0, i), aux2);
		mpf_sub(a, aux1, aux2);
	}
	mpf_ui_div(a, 1, a);
	mpf_printf("%F\n", a);
	mpf_clear(a);
	mpf_clear(r);
	mpf_clear(s);
	mpf_clear(aux1);
	mpf_clear(aux2);
	return 0;
}
