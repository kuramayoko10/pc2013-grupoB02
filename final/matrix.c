#ifndef MATRIX_H
#define MATRIX_H

struct smatrix
{
	unsigned order;
	float **values;
};

struct matrix *smatrix_new(unsigned order);
float smatrix_at(struct smatrix *matrix, unsigned i, unsigned j);
void smatrix_free(struct smatrix *matrix);

#endif
