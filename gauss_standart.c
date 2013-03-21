/*
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include "common.h"

// CONSTANTES
int DIGITS = 10000000;
float BITS_PER_DIGIT = 4;
int NUM_ITERATIONS = 25;

typedef struct
{
    mpf_t prevValue;    // Valor da variavel na iteracao anterior
    mpf_t curValue;     // Valor da variavel na iteracao atual
}BigNumber;


void initBigNumber(BigNumber *bn)
{
    mpf_init(bn->prevValue);
    mpf_init(bn->curValue);
}

void metodoGaussLegendre(double limInf, double limSup, int numPontos)
{
    int ret = 1;
    int iter = 0;
    BigNumber a;
    BigNumber b;
    BigNumber t;
    BigNumber p;
    BigNumber PI;

    // Set a precisao padrao para os calculos
    // Baseado no algoritimo de Chudnovsky usando GMP
    mpf_set_default_prec((long)(DIGITS*BITS_PER_DIGIT+16));
    
    // Initialize the BigNumbers
    initBigNumber(&a);
    initBigNumber(&b);
    initBigNumber(&t);
    initBigNumber(&p);
    initBigNumber(&PI);
    
    // Set values for iteration 0
    mpf_set_d(a.prevValue, 1.0);
    mpf_set_d(b.prevValue, 1.0/sqrt(2.0));
    mpf_set_d(t.prevValue, 1.0/4.0);
    mpf_set_d(p.prevValue, 1.0);
    
    while(iter != NUM_ITERATIONS)
    {
        // an+1 = (an+bn)/2
        mpf_add(a.curValue, a.prevValue, b.prevValue);
        mpf_div_ui(a.curValue, a.curValue, 2);
        
        // bn+1 = sqrt(an*bn)
        mpf_mul(b.curValue, b.prevValue, a.prevValue);
        mpf_sqrt(b.curValue, b.curValue);
        
        // tn+1 = tn - pn(an - an+1)^2
        mpf_sub(t.curValue, a.prevValue, a.curValue);
        mpf_pow_ui(t.curValue, t.curValue, 2);
        mpf_mul(t.curValue, t.curValue, p.prevValue);
        mpf_sub(t.curValue, t.prevValue, t.curValue);
        
        // pn+1 = 2*pn
        mpf_mul_ui(p.curValue, p.prevValue, 2);
        
        // PI = (an + bn)^2 / 4*tn
        mpf_add(PI.curValue, a.prevValue, b.prevValue);
        mpf_pow_ui(PI.curValue, PI.curValue, 2);
        mpf_div(PI.curValue, PI.curValue, t.prevValue);
        mpf_div_ui(PI.curValue, PI.curValue, 4);
        
        // Reseta os valores para proxima iteracao
        mpf_set(a.prevValue, a.curValue);
        mpf_set(b.prevValue, b.curValue);
        mpf_set(t.prevValue, t.curValue);
        mpf_set(p.prevValue, p.curValue);
        mpf_set(PI.prevValue, PI.curValue);
        
        iter++;
        
        mpf_out_str(stdout, 10, DIGITS/(NUM_ITERATIONS*10-iter), PI.curValue);
        printf("\n");
    }
    
    //ret = mpf_out_str(stdout, 10, DIGITS, PI.curValue);
    //gmp_printf("%1.*Ff\n", 100000, PI);
    
    if(ret == 0)
        printf("ERROR\n");
    
    //printf("ACABOU!\n");
}


int main(int argc, const char * argv[])
{
    
    metodoGaussLegendre(1.0, 1.0/sqrt(2.0), 10);
    
    return 0;
}
*/
