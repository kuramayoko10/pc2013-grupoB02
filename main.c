
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void metodoGaussLegendre(double limInf, double limSup, int numPontos)
{
    long double a = 1.0;
    long double b = 1.0 / sqrt(2);
    long double t = 1.0/4.0;
    long double x = 1;
    
    while((a-b) > 0.000000001)
    {
        long double y = a;
        a = (a+b)/2;
        b = sqrt(b*y);
        t -= x * pow(y-a, 2);
        x *= 2.0;
        
        printf("PI: %1.100Lf\n", pow(a+b, 2)/(4*t));
        //a = y;
    }
    
}


int main(int argc, const char * argv[])
{
    
    metodoGaussLegendre(1.0, 1.0/sqrt(2.0), 10);
    
    return 0;
}

