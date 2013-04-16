#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<gmp.h>

//#define PRECISION 10000000000 /*em bits*/

int DIGITS = 1000000;
float BITS_PER_DIGIT = 3.32192809488736234787f;

int main(void){
	double x, y;
	mpf_set_default_prec((long)(DIGITS*BITS_PER_DIGIT+16));
	/*declarações de variaveis*/
	mpf_t PI, dentroCircunferencia, TOTAL, cont;

	/*inicializações padroes*/
	mpf_init (PI);	
	mpf_init (TOTAL);
	mpf_init (dentroCircunferencia);	
	mpf_init (cont);
	mpf_set_d(dentroCircunferencia, 0.0);
	mpf_set_d(cont, 0.0);
	mpf_set_d(TOTAL, 1000000000.0);
 
	srand(time(NULL));
	for(;mpf_cmp (TOTAL, cont); mpf_add_ui(cont, cont, 1)){
		x = drand48();
		y = drand48();
		if((pow(x, 2) + pow(y,2)) <= 1)
			mpf_add_ui (dentroCircunferencia, dentroCircunferencia, 1);		
	}
	mpf_div (PI, dentroCircunferencia, TOTAL);
	mpf_mul_ui (PI, PI, 4);
	mpf_out_str(stdout, 10, DIGITS, PI);
	printf("\n");

	mpf_clear (PI);
	mpf_clear (TOTAL);
	mpf_clear (dentroCircunferencia);
	mpf_clear (cont);
	return 1;
}


