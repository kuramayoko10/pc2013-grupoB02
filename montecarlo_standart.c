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
	int geradorX, geradorY;
	mpf_set_default_prec((long)(DIGITS*BITS_PER_DIGIT+16));
	/*declarações de variaveis*/
	mpz_t seed;
	//mpf_t x, y, z, PI, dentroCircunferencia, TOTAL, cont;
	mpf_t z, PI, dentroCircunferencia, TOTAL, cont;
	gmp_randstate_t randState;

	/*inicializações padroes*/
	//mpf_init2 (x, PRECISION);	
	//mpf_init2 (y, PRECISION);	
	mpf_init (z);	
	mpf_init (PI);	
	mpf_init (TOTAL);
	mpf_init (dentroCircunferencia);	
	mpf_init (cont);
	mpz_init (seed);			
	mpf_set_d(dentroCircunferencia, 0.0);
	mpf_set_d(cont, 0.0);
	mpf_set_d(TOTAL, 1000000000.0);
 
	srand(time(NULL));
	for(;mpf_cmp (TOTAL, cont); mpf_add_ui(cont, cont, 1)){
		/*gerador de números randomicos, caso deseje usar a GMP, porém deu para perceber que nao precisa usar ela para gerar os numeros randomicos, pq so demora mais, e o resultado nao se altera, pq com milhares de iterações, os numeros nao ficam tao diferentes, entao um random da biblioteca LIB ja eh suficiente*//*
		geradorX = rand();	
		mpz_set_d(seed, geradorX);
		gmp_randinit_mt (randState);
		gmp_randseed(randState, seed);
		mpf_urandomb (x, randState, PRECISION);

		geradorY = rand();
		mpz_set_d(seed, geradorY);
		gmp_randinit_mt (randState);
		gmp_randseed(randState, seed);
		mpf_urandomb (y, randState, PRECISION);

		mpf_pow_ui (x, x, 2);	
		mpf_pow_ui (y, y, 2);	

		mpf_add (z, x, y);
		if(!mpf_cmp_ui (z, 1))	*/
		x = drand48();
		y = drand48();
		if((pow(x, 2) + pow(y,2)) <= 1)
			mpf_add_ui (dentroCircunferencia, dentroCircunferencia, 1);		
	}
	mpf_div (PI, dentroCircunferencia, TOTAL);
	mpf_mul_ui (PI, PI, 4);
	mpf_out_str(stdout, 10, DIGITS, PI);
	printf("\n");
	//mpf_clear (x);
	//mpf_clear (y);
	//mpf_clear (z);
	mpf_clear (PI);
	mpf_clear (TOTAL);
	mpf_clear (dentroCircunferencia);
	mpf_clear (cont);
	mpz_clear (seed);
	return 1;
}


