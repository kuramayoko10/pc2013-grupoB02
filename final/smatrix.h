#ifndef MATRIX_H
#define MATRIX_H

//Define para obter o conteudo da matriz, organizada na forma de vetor
//Deixa o codigo mais legivel e evita erros no acesso ao vetor
#define smatrix_at(A, i, j) A[i*order+j]

//Metodos para alocar/limpar/setar matrizes e vetores
//Utilizado na leitura dos dados do arquivo de entrada
float *smatrix_new(unsigned order);
void smatrix_free(float *matrix);
void smatrix_set(float *matrix, unsigned i, unsigned j, float val);
float * vector_new(unsigned size);
void vector_free(float *vec);
void vector_print(float *vec);
float test_row(float *A, float *x);

#endif

