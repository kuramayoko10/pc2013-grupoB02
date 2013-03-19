
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>


void metodoGaussLegendre(double limInf, double limSup, int numPontos)
{
    mpf_t a;
    mpf_t b;
    mpf_t t;
    mpf_t x;
    mpf_t PI;
    mpf_t const2;
    mpf_t const4;
    mpf_t tmp;
    mpf_t tmp2;
    
    int ret = 1;
    int done = 0;
    int iter = 0;
    
    mpf_set_default_prec(256);
    
    // Initialize Floats
    mpf_init(PI);
    mpf_init(a);
    mpf_init(b);
    mpf_init(t);
    mpf_init(x);
    mpf_init(tmp);
    mpf_init(tmp2);
    mpf_init(const2);
    mpf_init(const4);
    
    // Set values to the variables
    mpf_set_d(a, 1.0);
    mpf_set_d(b, 1.0 / sqrt(2.0));
    mpf_set_d(t, 1.0/4.0);
    mpf_set_d(x, 1.0);
    mpf_set_d(const2, 2.0);
    mpf_set_d(const4, 4.0);
    
    mpf_t y;
    mpf_init(y);
    
    //(a-b) > 0.0000001
    while(1)
    {
        //printf("iter x\n");
        //printf("\n\n\n\n\n\n\n\n\n\n\n");
        
        /*mpf_sub(tmp, a, b);
        if(mpf_cmp_d(tmp, 0.0000001) < 0)
            break;*/
        
        if(iter == 10)
            break;
        
        //y = a
        mpf_set(y, a);
        
        //a = (a+b)/2;
        mpf_add(a, a, b);
        mpf_div(a, a, const2);
        
        //b = sqrt(b*y);
        mpf_mul(b, b, y);
        mpf_sqrt(b, b);
        
        //t -= x * pow(y-a, 2);
        mpf_sub(tmp2, y, a);
        mpf_pow_ui(tmp, tmp2, 2);
        mpf_mul(tmp, tmp, x);
        mpf_sub(t, t, tmp);
        
        //x *= 2.0;
        mpf_mul(x, x, const2);
        
        //printf("PI: %1.100Lf\n", pow(a+b, 2)/(4*t));
        mpf_mul(tmp, t, const4);
        mpf_add(tmp2, a, b);
        mpf_pow_ui(tmp2, tmp2, 2);
        mpf_div(PI, tmp2, tmp);
        
        //ret = mpf_out_str(stdout, 10, 1000000, PI);
        gmp_printf("%.1000Ff\n", PI);
        
        if(ret == 0)
            printf("ERROR\n");
        
        printf("\n");
        iter++;
    }
    
    //printf("ACABOU!\n");
}


int main(int argc, const char * argv[])
{
    
    metodoGaussLegendre(1.0, 1.0/sqrt(2.0), 10);
    
    return 0;
}

