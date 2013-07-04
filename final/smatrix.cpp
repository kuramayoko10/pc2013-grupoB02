#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <stdio.h>
#include "smatrix.h" 
#include "extern.h"


//Aloca o espaco desejado para uma matriz, na forma de vetor
float *smatrix_new(unsigned order)
{
	float *matrix;
	matrix = (float *) malloc(order*order*sizeof(float *));	
	if (matrix == NULL)
	{
		abort();
		return NULL;
	}
	return matrix;
}

//Libera o espaco alocado pela matriz
void smatrix_free(float *matrix)
{
	free(matrix);
}

//Seta um valor dentro da matriz
void smatrix_set(float *matrix, unsigned i, unsigned j, float val)
{
	matrix[i*order+j] = val;
}

//Aloca o espaco desejado para um vetor
float *vector_new(unsigned size)
{
	float *vec;
	vec = (float *) malloc(size*sizeof(float));
	if (vec == NULL)
	{
		abort();
		return NULL;
	}
	return vec;
}

//Libera o espaco alocado para o vetor
void vector_free(float *vec)
{
	free(vec);
}

//Imprime o vetor
void vector_print(float *vec)
{
	unsigned i;
	printf("Vector at %p.\n", (void *) vec);
	for (i=0; i<order; ++i)
	{
		printf("At vec[%u] = %f.\n", i, vec[i]);
	}
}

//Obtem o resultado da linha que deseja-se avaliar (Row_test)
float test_row(float *A, float *x)
{
	unsigned i;
	float sum = 0;
	for (i=0; i<order; ++i)
	{
		sum += (smatrix_at(A, row_test, i))*(x[i]);
	}
	return sum;
}

