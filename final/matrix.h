#include <stdlib.h>
#include "smatrix.h"

struct matrix *smatrix_new(unsigned order)
{
	struct matrix *matrix;
	unsigned i;
	matrix->order = order;
	matrix->values = malloc(order*sizeof(float *));	
	if (matrix->values == NULL)
		goto pp_error;
	for (i=0; i<order; ++i)
	{
		matrix->values[i] = malloc(order*sizeof(float));
		if (matrix->values[i] == NULL)
			goto p_error;
	}
	return matrix;
p_error:
	for (--i;i>=0; --i)
		free(matrix->values[i];	
pp_error:
	free(matrix->values);
	return NULL;
}

float smatrix_at(struct smatrix *matrix, unsigned i, unsigned j)
{
	assert(i>=0 && i<matrix->order && j>=0 && j<matrix->order)
	return matrix[i][j];
}

void smatrix_free(struct smatrix *matrix)
{
	unsigned i;
	for (i=0; i<matrix->order; ++i)
		free(matrix->values[i]);
	free(matrix->values);
}
