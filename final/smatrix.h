#ifndef MATRIX_H
#define MATRIX_H

float ** smatrix_new(unsigned order);
void smatrix_free(float **matrix);
float * vector_new(unsigned size);
void vector_free(float *vec);
float test_row(float **A, float *x);

#endif
