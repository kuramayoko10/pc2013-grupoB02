#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <stdio.h>
#include "smatrix.h" 
#include "extern.h"


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

void smatrix_free(float *matrix)
{
	free(matrix);
}

void smatrix_set(float *matrix, unsigned i, unsigned j, float val)
{
	matrix[i*order+j] = val;
}
	
float smatrix_at(float *matrix, unsigned i, unsigned j)
{
	return matrix[i*order+j];
}

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

void vector_free(float *vec)
{
	free(vec);
}

void vector_print(float *vec)
{
	unsigned i;
	printf("Vector at %p.\n", (void *) vec);
	for (i=0; i<order; ++i)
	{
		printf("At vec[%u] = %f.\n", i, vec[i]);
	}
}

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

