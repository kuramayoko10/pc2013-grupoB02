#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "extern.h"
#include "smatrix.h"

//Metodo Solve no host (cpu)
//Metodo chamado no main()
//
//@param A - matriz de coeficientes do sistema linear
//@param b - vetor com os valores `a direita da igualdade
//@return - o vetor com a solucao para todas as incognitas 'x'
float *solve(float *A, float *b)
{
	unsigned i, j, it=0;
	float sum, maxdiff=0.0, maxx=0.0;
	float *x, *oldx;
    
	x = vector_new(order);
	oldx = vector_new(order);
    
    //Inicializa o x
	for (i=0; i<order; ++i)
		x[i] = 1.0;
    
    //Enquanto o erro OU o numero de iteracoes nao atingir o limite esperado
    //continua realizando as iteracoes
	while (it<it_num || maxdiff/maxx > err)
	{
        //Para cada linha da matriz
		for (i=0; i<order; i++)
		{
			sum = b[i];
			maxdiff = 0.0;
			maxx = 0.0;
            
            //Equacao (b - R*x)
			for (j=0; j<order; ++j)
				sum -= (i!=j)?smatrix_at(A, i, j)*oldx[j]:0;
            
            //Equacao - D*(b - R*x)
			x[i] = sum/smatrix_at(A, i, i);
            
            //Calculo do erro acumulado
			for (j=0; j<order; ++j)
			{
				float aux = abs(x[j]);
				if (aux>maxx)
					maxx = aux;	
				aux = abs(x[j]-oldx[j]);
				if (aux>maxdiff)
					maxdiff = aux;
			}
		}
        
        //Guarda o valor obtido nesta iteracao
		for (i=0; i<order; i++)
			oldx[i] = x[i];
        
        //Aumenta o contador do numero de iteracoes
		++it;
	}
    
    //Libera a memoria
	vector_free(oldx);
	it_num = it;
    
	return x;
}
