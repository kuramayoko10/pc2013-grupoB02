#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include "smatrix.h" 
#include "extern.h"


float **smatrix_new(unsigned order)
{
	float **matrix;
	unsigned i;
	matrix = malloc(order*sizeof(float *));	
	if (matrix == NULL)
		goto pp_error;
	for (i=0; i<order; ++i)
	{
		matrix[i] = malloc(order*sizeof(float));
		if (matrix[i] == NULL)
			goto p_error;
	}
	return matrix;
p_error:
	for (--i;i!=UINT_MAX; --i)
		free(matrix[i]);	
	free(matrix);
pp_error:
	abort();
	return NULL;
}

void smatrix_free(float **matrix)
{
	unsigned i;
	for (i=0; i<order; ++i)
		free(matrix[i]);
	free(matrix);
}

float *vector_new(unsigned size)
{
	float *vec;
	vec = malloc(size*sizeof(float));
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

float test_row(float **A, float *x)
{
	unsigned i;
	float sum = 0;
	for (i=0; i<order; ++i)
	{
		sum += (A[row_test][i])*(x[i]);
	}
	return sum;
}
