#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "extern.h"
#include "smatrix.h"

float *solve(float *A, float *b)
{
	unsigned i, j, it=0;
	float sum, maxdiff=0.0, maxx=0.0;
	float *x, *oldx;
	x = vector_new(order);
	oldx = vector_new(order);
	for (i=0; i<order; ++i)
		x[i] = 1.0;
	while (it<it_num || maxdiff/maxx > err)
	{
		for (i=0; i<order; i++)
		{
			sum = b[i];
			maxdiff = 0.0;
			maxx = 0.0;
			for (j=0; j<order; ++j)
				sum -= (i!=j)?smatrix_at(A, i, j)*oldx[j]:0;
			x[i] = sum/smatrix_at(A, i, i);
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
		for (i=0; i<order; i++)
			oldx[i] = x[i];
		++it;
	}
	vector_free(oldx);
	it_num = it;
	return x;
}
