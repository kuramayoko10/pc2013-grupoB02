#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<gmp.h>
#include <pthread.h>


#define NUMTHREADS 10	/* Total de pontos da thread */

/*funcao que calcula pontos fora e dentro da circunferencia*/
void *calcula (void *param);

/*Digitos de precisao*/
int DIGITS = 1000000;
float BITS_PER_DIGIT = 3.32192809488736234787f;

/*variaveis globais que serao usados entre as threads*/
int dentroCircunferenciaParcial[NUMTHREADS];
int TotalParcial[NUMTHREADS];


int main(void){

	/*inicializa as variaveis*/
	int cont;
	mpf_set_default_prec((long)(DIGITS*BITS_PER_DIGIT+16));
	mpf_t PI, dentroCircunferencia, TOTAL;
	pthread_t tID[NUMTHREADS];  // ID das threads

	mpf_init(TOTAL);
	mpf_init(dentroCircunferencia);
	mpf_init(PI);
	mpf_set_d(dentroCircunferencia, 0.0);
	mpf_set_d(TOTAL, 0.0);	

	// Para todas as threads, cria a i-esima thread      
	for (cont = 0; cont< NUMTHREADS ; cont++)
		pthread_create (&tID[cont], NULL, calcula, &cont);   	
	// Para cada thread, espera que as threads terminem 
	for (cont = 0; cont< NUMTHREADS ; cont++)
		pthread_join (tID[cont], NULL);
    for (cont = 0; cont< NUMTHREADS ; cont++){
          mpf_add_ui(TOTAL, TOTAL, TotalParcial[cont]);
          mpf_add_ui(dentroCircunferencia, dentroCircunferencia, dentroCircunferenciaParcial[cont]);
    }	
	mpf_div (PI, dentroCircunferencia, TOTAL);
	mpf_mul_ui (PI, PI, 4);
	mpf_out_str(stdout, 10, DIGITS, PI);
	printf("\n");	

	mpf_clear (PI);
	mpf_clear (TOTAL);
	mpf_clear (dentroCircunferencia);
	return 1;
}

/*cada Thread calcula um numero de vezes quantos pontos ficam dentro da circunferencia. No final ele junta todas essas somas*/
void *calcula (void *param) {
	int i;
	int thrNum = *((int *)param); // O número da thread ()
	double x,y;

	for (i = 0; i<1000000; i++){
		x = drand48();
		y = drand48();
		if((pow(x, 2) + pow(y,2)) <= 1)
			dentroCircunferenciaParcial[thrNum]++;
		TotalParcial[thrNum]++;
	}
	pthread_exit(0);
}
