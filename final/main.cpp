#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "smatrix.h"

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
	if (argc != 2)
		return FAILURE;
	strcat(path, argv[1]);
	file = fopen(path, "r");
	if (file == NULL)
	{
		printf("Failed opening file %s, call the program passing a number as parameter. Ex: './solver 500'.\n", path);
		return FAILURE;
	}
	fscanf(file, "%u", &order);
	fscanf(file, "%u", &row_test);
	fscanf(file, "%f", &err);
	fscanf(file, "%u", &it_num);
	A = smatrix_new(order);
	b = vector_new(order);
	for (i=0; i<order; ++i)
	{
		for (j=0; j<order; ++j)
		{
			float val;
			fscanf(file, "%f", &val);
			smatrix_set(A, i, j, val/10.0);
		}	
	}
	for (i=0; i<order; ++i)
	{
		float val;
		fscanf(file, "%f", &val);
		&b[i]=val/10.0;
	}
	begin = clock();
	res = solve(A, b);
	end = clock();
	TIME_DIFF(begin, end);
	test_row(A, res);
	printf("Iterations: %u\n", it_num);
	printf("RowTest: %d, [%f] =? %f\n", row_test, test_row(A, res), 
			b[row_test]);
	vector_free(b);
	vector_free(res);
	smatrix_free(A);
	fclose(file);
	return SUCCESS;
}
