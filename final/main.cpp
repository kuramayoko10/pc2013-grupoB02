#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "smatrix.h"

//Esse valor foi utilizado na tentativa de reduzir o tamanho e precisao dos elementos lidos do arquivo de entrada
//Essa tentativa se deu para evitar o float overflow que ocorre nos casos 3000 e 4000
//Resetado para 1.0 na versao final
#define RATIO 1.0 

//Funcao que implementa o metodo de Jacobi-Richardson
//A definicao do metodo existe tanto no cpu_solve quanto no gpu_solve
float *solve(float *, float *);

unsigned order;
unsigned row_test;
unsigned it_num;
float err;

int main(int argc, char **argv)
{
	unsigned i, j;
	char path[20]="matriz";
	FILE *file;
	float *A;
	float *b, *res;
	clock_t begin, end;
    
    //Se o arquivo de entrada nao for providenciado, para a execucao
	if (argc != 2)
		return FAILURE;
    
	strcat(path, argv[1]);
	file = fopen(path, "r");
	if (file == NULL)
	{
		printf("Failed opening file %s, call the program passing a number as parameter. Ex: './solver 500'.\n", path);
		return FAILURE;
	}
    //Le as configuracoes do arquivo
	fscanf(file, "%u", &order);
	fscanf(file, "%u", &row_test);
	fscanf(file, "%f", &err);
	fscanf(file, "%u", &it_num);
    
    //Aloca o espaco necessario
	A = smatrix_new(order);
	b = vector_new(order);
    
    //Leitura dos dados
	for (i=0; i<order; ++i)
	{
		for (j=0; j<order; ++j)
		{
			float val;
			fscanf(file, "%f", &val);
			smatrix_set(A, i, j, val/RATIO);
		}	
	}
	for (i=0; i<order; ++i)
	{
		float val;
		fscanf(file, "%f", &val);
		b[i]=val/RATIO;
	}

    //Chama a solucao e armazena no vetor res
	res = solve(A, b);

    //Saida final
	test_row(A, res);
	printf("Iterations: %u\n", it_num);
	printf("RowTest: %d, [%f] =? %f\n", row_test, test_row(A, res), 
			b[row_test]);
    
    //Limpeza de memoria
	vector_free(b);
	vector_free(res);
	smatrix_free(A);
	fclose(file);
    
	return SUCCESS;
}
