#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<gmp.h>
#include <pthread.h>

#define NUMTHREADS 5	/* Total de pontos da thread */
#define NUMIT 2000000 /*Numero de iteracoes por thread*/

/*funcao que calcula pontos fora e dentro da circunferencia*/
void *calcula (void *param);

/*Digitos de precisao*/
int DIGITS = 1000000;
float BITS_PER_DIGIT = 3.32192809488736234787f;

/*variaveis globais que serao usados entre as threads*/
unsigned long int dentroCircunferenciaParcial[NUMTHREADS];
unsigned long int TOTALParcial[NUMTHREADS];

int main(void){
	/*inicializa as variaveis*/
	int cont, numThread[NUMTHREADS];
	mpf_set_default_prec((long)(DIGITS*BITS_PER_DIGIT+16));
	mpf_t PI, dentroCircunferencia, TOTAL;
	pthread_t tID[NUMTHREADS];  // ID das threads

	mpf_init(PI);
	mpf_init_set_d(dentroCircunferencia, 0.0);
	mpf_init_set_d(TOTAL, 0.0);	
	srand48(time(NULL));

	// Para todas as threads, cria a i-esima thread      
	for (cont = 0; cont< NUMTHREADS ; cont++){
		numThread[cont] = cont;
		pthread_create (&tID[cont], NULL, calcula, &numThread[cont]);   	
	}
	// Para cada thread, espera que as threads terminem 
	for (cont = 0; cont< NUMTHREADS ; cont++)
		pthread_join (tID[cont], NULL);

	//Para cada thread, soma a sua parcela na conta total
    for (cont = 0; cont< NUMTHREADS ; cont++){
          mpf_add_ui(TOTAL, TOTAL, TOTALParcial[cont]);
          mpf_add_ui(dentroCircunferencia, dentroCircunferencia, dentroCircunferenciaParcial[cont]);
	}

	mpf_div (PI, dentroCircunferencia, TOTAL);
	mpf_mul_ui (PI, PI, 4);
	mpf_out_str(stdout, 10, DIGITS, PI);
	printf("\n");	

	mpf_clear (PI);
	mpf_clear (TOTAL);
	mpf_clear (dentroCircunferencia);
	return 0;
}

/*cada Thread calcula um numero de vezes quantos pontos ficam dentro da circunferencia. No final ele junta todas essas somas*/
void *calcula (void *param) {
	int i=0;
	int thrNum = *((int *)param); // O número da thread ()
	double x,y;
	dentroCircunferenciaParcial[thrNum] = 0;
	TOTALParcial[thrNum] = 0;

	for (i = 0; i<NUMIT; i++){
		TOTALParcial[thrNum]++;
		x = drand48();
		y = drand48();
		if((pow(x, 2) + pow(y,2)) <= 1.0)
			dentroCircunferenciaParcial[thrNum]++;
	}
	pthread_exit(0);
}
