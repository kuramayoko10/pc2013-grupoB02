
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <pthread.h>

// CONSTANTES
int DIGITS = 10000000;
float BITS_PER_DIGIT = 4;
int NUM_ITERATIONS = 24;
int NUM_OPERATIONS = 5;

// Estrutura BigNumber que contem os valores da iteracao atual e da anterior
typedef struct
{
    mpf_t prevValue;    // Valor da variavel na iteracao anterior
    mpf_t curValue;     // Valor da variavel na iteracao atual
}BigNumber;

// Estrutura GaussLegendre para passar por parametro para as threads
typedef struct
{
    BigNumber a;
    BigNumber b;
    BigNumber t;
    BigNumber p;
    BigNumber PI;
}GaussLegendre;


void metodoGaussLegendre(double limitSup, double limitInf, double tValue, int nPoints);
void initBigNumber(BigNumber *bn);
void* calculatePI(void *gl);
void* updateA(void *gl);
void* updateB(void *gl);
void* updateP(void *gl);
void* updateT(void *gl);


int main(int argc, const char * argv[])
{
    // Executa o algoritimo de GaussLegendre para calculo do PI
    metodoGaussLegendre(1.0, 0.5, 0.25, 1);
    
    return 0;
}


void metodoGaussLegendre(double limitSup, double limitInf, double tValue, int nPoints)
{
    int iter = 0;
    int i = 0;
    void *status;
    
    GaussLegendre *gl;
    pthread_t threads[NUM_OPERATIONS];
    
    // Set a precisao padrao para os calculos
    // Baseado no algoritimo de Chudnovsky usando GMP
    mpf_set_default_prec((long)(DIGITS*BITS_PER_DIGIT+16));
    
    // Inicia as estruturas BigNumber
    gl = (GaussLegendre*)malloc(sizeof(GaussLegendre));
    initBigNumber(&gl->a);
    initBigNumber(&gl->b);
    initBigNumber(&gl->t);
    initBigNumber(&gl->p);
    initBigNumber(&gl->PI);
    
    // Set valores para a iteracao 0
    mpf_set_d(gl->a.prevValue, limitSup);
    mpf_set_d(gl->b.prevValue, limitInf);
    mpf_sqrt(gl->b.prevValue, gl->b.prevValue);
    mpf_set_d(gl->t.prevValue, tValue);
    mpf_set_d(gl->p.prevValue, nPoints);
    
    while(iter != NUM_ITERATIONS)
    {
        // Cria uma thread para cada operacao independente que pode ser executada em paralelo
        pthread_create(&threads[0], NULL, calculatePI, (void*)gl);
        pthread_create(&threads[1], NULL, updateA, (void*)gl);
        pthread_create(&threads[2], NULL, updateB, (void*)gl);
        pthread_create(&threads[3], NULL, updateP, (void*)gl);
        
        // A operacao de atualizacao de Tn depende de An+1
        pthread_join(threads[1], &status);
        pthread_create(&threads[4], NULL, updateT, (void*)gl);
        
        //Aguarda todas as threads terminarem antes de continuar para proxima iteracao
        for(i = 0; i < NUM_OPERATIONS; i++)
            pthread_join(threads[i], &status);
        
        // Atualiza os valores para proxima iteracao
        mpf_set(gl->a.prevValue, gl->a.curValue);
        mpf_set(gl->b.prevValue, gl->b.curValue);
        mpf_set(gl->t.prevValue, gl->t.curValue);
        mpf_set(gl->p.prevValue, gl->p.curValue);
        mpf_set(gl->PI.prevValue, gl->PI.curValue);
        
        iter++;
        
        // Imprime o resultado a cada iteracao
        //mpf_out_str(stdout, 10, DIGITS, PI.curValue);
        //printf("\n\n");
    }
    
    // Imprime o resultado final
    mpf_out_str(stdout, 10, DIGITS, gl->PI.curValue);
}

void initBigNumber(BigNumber *bn)
{
    mpf_init(bn->prevValue);
    mpf_init(bn->curValue);
}

void* calculatePI(void *gl)
{
    GaussLegendre *gauss = (GaussLegendre*)gl;
    
    // PI = (an + bn)^2 / (4*tn)
    mpf_add(gauss->PI.curValue, gauss->a.prevValue, gauss->b.prevValue);
    mpf_pow_ui(gauss->PI.curValue, gauss->PI.curValue, 2);
    mpf_div(gauss->PI.curValue, gauss->PI.curValue, gauss->t.prevValue);
    mpf_div_ui(gauss->PI.curValue, gauss->PI.curValue, 4);
    
    return (void*)0;
}

void* updateA(void *gl)
{
    GaussLegendre *gauss = (GaussLegendre*)gl;
    
    // an+1 = (an+bn)/2
    mpf_add(gauss->a.curValue, gauss->a.prevValue, gauss->b.prevValue);
    mpf_div_ui(gauss->a.curValue, gauss->a.curValue, 2);
    
    return (void*)0;
}

void* updateB(void *gl)
{
    GaussLegendre *gauss = (GaussLegendre*)gl;
    
    // bn+1 = sqrt(an*bn)
    mpf_mul(gauss->b.curValue, gauss->b.prevValue, gauss->a.prevValue);
    mpf_sqrt(gauss->b.curValue, gauss->b.curValue);
    
    return (void*)0;
}

void* updateP(void *gl)
{
    GaussLegendre *gauss = (GaussLegendre*)gl;
    
    // pn+1 = 2*pn
    mpf_mul_ui(gauss->p.curValue, gauss->p.prevValue, 2);
    
    return (void*)0;
}

void* updateT(void *gl)
{
    GaussLegendre *gauss = (GaussLegendre*)gl;
    
    // tn+1 = tn - pn(an - an+1)^2
    mpf_sub(gauss->t.curValue, gauss->a.prevValue, gauss->a.curValue);
    mpf_pow_ui(gauss->t.curValue, gauss->t.curValue, 2);
    mpf_mul(gauss->t.curValue, gauss->t.curValue, gauss->p.prevValue);
    mpf_sub(gauss->t.curValue, gauss->t.prevValue, gauss->t.curValue);
    
    return (void*)0;
}

