
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>

// CONSTANTES
int DIGITS = 10000000;
float BITS_PER_DIGIT = 4;
int NUM_ITERATIONS = 24;

// Estrutura BigNumber que contem os valores da iteracao atual e da anterior
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

void metodoGaussLegendre(double limitSup, double limitInf, double tValue, int nPoints)
{
    int iter = 0;
    BigNumber a;
    BigNumber b;
    BigNumber t;
    BigNumber p;
    BigNumber PI;
    //mpf_t condition;
    //mpf_t precision;

    // Seta precisao padrao para os calculos
    // Baseado no algoritimo de Chudnovsky usando GMP
    mpf_set_default_prec((long)(DIGITS*BITS_PER_DIGIT+16));
    
    // Initialize the BigNumbers
    initBigNumber(&a);
    initBigNumber(&b);
    initBigNumber(&t);
    initBigNumber(&p);
    initBigNumber(&PI);
    //mpf_init(condition);
    //mpf_init(precision);
    
    // Set values for iteration 0
    mpf_set_d(a.prevValue, limitSup);
    mpf_set_d(b.prevValue, limitInf);
    mpf_sqrt(b.prevValue, b.prevValue);
    mpf_set_d(t.prevValue, tValue);
    mpf_set_d(p.prevValue, nPoints);
    //mpf_set_d(condition, 0);
    //mpf_set_d(precision, DIGITS*BITS_PER_DIGIT+16);
    //mpf_ui_div(precision, 1, precision);
    
    //while(1)
    while(iter != NUM_ITERATIONS)
    {
        
        // Verificacao da condicao de numero de casas decimais atingidas
        //mpf_sub(condition, a.prevValue, b.prevValue);
        //if(mpf_cmp(condition, precision) <= 0)
        //{
        //    break;
        //}
        
        // PI = (an + bn)^2 / (4*tn)
        mpf_add(PI.curValue, a.prevValue, b.prevValue);
        mpf_pow_ui(PI.curValue, PI.curValue, 2);
        mpf_div(PI.curValue, PI.curValue, t.prevValue);
        mpf_div_ui(PI.curValue, PI.curValue, 4);
        
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
        
        // Atualiza os valores para proxima iteracao
        mpf_set(a.prevValue, a.curValue);
        mpf_set(b.prevValue, b.curValue);
        mpf_set(t.prevValue, t.curValue);
        mpf_set(p.prevValue, p.curValue);
        mpf_set(PI.prevValue, PI.curValue);
        
        // Imprime o resultado a cada iteracao
        // mpf_out_str(stdout, 10, DIGITS, PI.curValue);
        
        iter++;
    }
    
    //printf("iter=%d\n", iter);
    
    // Imprime o resultado final
    mpf_out_str(stdout, 10, DIGITS, PI.curValue);
}


int main(int argc, const char * argv[])
{
    // Executa o algoritimo de GaussLegendre para calculo do PI
    metodoGaussLegendre(1.0, 0.5, 1.0/4.0, 1);
    
    return 0;
}
