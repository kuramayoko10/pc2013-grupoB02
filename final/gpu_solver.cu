/*
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

//Kernel - metodo no dispositivo
__global__ void AddVectors(int *a, int *b, int count)
{
int uid = blockIdx.x * blockDim.x + threadIdx.x;

if(uid < count)
{
a[uid] = a[uid] + b[uid]
}
}

__global__ void SubtractVectors(int *b, int *rx, int count)
{
int uid = blockIdx.x * blockDim.x + threadIdx.x;

if(uid < count)
{
rx[uid] = b[uid] - rx[uid]
}
}

__global__ void MultiplyVectors(int *d, int *rxb, int count)
{
int uid = blockIdx.x * blockDim.x + threadIdx.x;

if(id < count)
{
rxb[uid] = rxb[uid] * d[uid];
}
}

//Cada thread trata de uma linha da matrix/vetor
__global__ void MultiplyMatrixVector(int *r, int *x, int numCols, int count)
{
int uid = blockIdx.x * blockDim.x + threadIdx.x;
int i;

if(uid < count)
{
for(i = 0; i < numCols; i++)
rxb[uid*numCols] += rxb[uid*numCols+i] * x[uid];
}
}

int main(int argc, char *argv[])
{


return 0;
}*/

#include <cuda.h>
#include <assert.h>
#include "extern.h"
#include "smatrix.h"

#define check(X) assert(X==cudaSuccess)

__global__ void kernel(float* A, float *b, float *x, float* oldx, unsigned err, unsigned *it, unsigned it_max) 
{ 
	unsigned idx = threadIdx.x + blockDim.x * blockIdx.x; 
	unsigned i;
	if(idx<order)
	{ 
		__shared__ float maxxdiff; 
		__shared__ float maxx; 
		maxxdiff=1; 
		maxx=1; 
		while((it<it_max)&&(maxxdiff/maxx)>err)
		{ 
			x[idx] = 0; 
			maxxdiff=0; 
			maxx=0; 
			for (i=0; i<order; ++i) 
			{ 
				x[idx]+=(i!=idx)?(smatrix_at(A, idx, i)*oldx[i]):0.0; 
			} 
			x[idx]=1/smatrix_at(A, idx, idx)*(b[idx]-x[idx]); 
			if(maxxdiff<abs(abs(x[idx])-abs(x[idx]))) 
				maxxdiff=abs(abs(x[idx])-abs(x[idx])); 
			if(maxx<abs(x[idx])) 
				maxx=abs(x[idx]); 
			oldx[idx]=x[idx]; 
			(*it)++;
		} 
	} 
}

float *solve(float *A, float *b)
{
	unsigned i, gpu_it, cpu_it=0;
	float *x;	
	float *gpu_A, *gpu_x, *gpu_b, *gpu_c;
	x = vector_new(order);
	for (i=0; i<order; ++i)
		x[i] = 1.0;
	check(cudaMalloc((void **)&gpu_A, order * order *sizeof(float)));
	check(cudaMalloc((void **)&gpu_b, order * sizeof(float)));
	check(cudaMalloc((void **)&gpu_x, sizeof(unsigned)));
	check(cudaMalloc((void **)&gpu_oldx, sizeof(unsigned)));
	check(cudaMalloc((void **)&gpu_it, sizeof(unsigned)));
	check(cudaMemcpy(gpu_A, A, order * order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_b, b, order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_x, x, order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_oldx, x, order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_it, &cpu_it, sizeof(unsigned), cudaMemcpyHostToDevice));
	kernel<<<order, 1>>>(gpu_A, gpu_x, gpu_b, order, gpu_it, it_num);	
	cudaMemcpy(x, gpu_x, order * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(it_num, gpu_it sizeof(unsigned), cudaMemcpyDeviceToHost);
	cudaFree(gpu_A);
	cudaFree(gpu_b);
	cudaFree(gpu_x);
	cudaFree(gpu_oldx);
	cudaFree(gpu_it);
	vector_print(x);
	return x;
}

