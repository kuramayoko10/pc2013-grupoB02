#ifndef MATRIX_H
#define MATRIX_H

float *smatrix_new(unsigned order);
void smatrix_free(float *matrix);
void smatrix_set(float *matrix, unsigned i, unsigned j, float val);
float smatrix_at(float *matrix, unsigned i, unsigned j);
float * vector_new(unsigned size);
void vector_free(float *vec);
float test_row(float *A, float *x);

#endif
