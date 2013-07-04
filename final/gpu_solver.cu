#include <cuda.h>
#include <assert.h>
#include <stdio.h>
#include "extern.h"
#include "smatrix.h"

#define THREADS_PER_BLOCK 250
#define check(X) 		\
{				\
	cudaError_t cerr = X;		\
	if (cerr != cudaSuccess){	\
		fprintf(stderr, "GPUassert:%s at line%d.\n", cudaGetErrorString(cerr), __LINE__);	\
		abort();			\
	}					\
}
#define my_abs(x) (x)>=0.0?(x):-1.0*(x)

__global__ void kernel(float* A, float *b, float *x, float* oldx, float err, unsigned order) 
{ 
	unsigned idx = threadIdx.x + blockDim.x * blockIdx.x; 
	unsigned i;
	float aux;
	if(idx<order)
	{ 
		__shared__ float maxxdiff; 
		__shared__ float maxx; 
		maxxdiff=1.0; 
		maxx=1.0; 
		while((maxxdiff/maxx)>err)
		{ 
			x[idx] = 0.0; 
			maxxdiff = 0.0; 
			maxx=0.0; 
			for (i=0; i<order; ++i) 
			{ 
				x[idx]+=(i!=idx)?(smatrix_at(A, idx, i)*oldx[i]):0.0; 
			} 
			x[idx] = (1.0/smatrix_at(A, idx, idx))*(b[idx]-x[idx]); 
			aux = my_abs(x[idx]);
			if (aux > maxx)
				maxx = aux;
			aux = abs(x[idx]-oldx[idx]);
			if (aux > maxxdiff)
				maxxdiff = aux;
			oldx[idx]=x[idx]; 
		} 
	} 
}

float *solve(float *A, float *b)
{
	unsigned i;
	float *x;	
	float *gpu_A, *gpu_b, *gpu_x, *gpu_oldx;
	x = vector_new(order);
	for (i=0; i<order; ++i)
		x[i] = 1.0;
	check(cudaMalloc((void **)&gpu_A, order * order * sizeof(float)));
	check(cudaMalloc((void **)&gpu_b, order * sizeof(float)));
	check(cudaMalloc((void **)&gpu_x, order * sizeof(float)));
	check(cudaMalloc((void **)&gpu_oldx, order * sizeof(float)));
	check(cudaMemcpy(gpu_A, A, order * order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_b, b, order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_x, x, order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_oldx, x, order * sizeof(float), cudaMemcpyHostToDevice));
	kernel<<<order/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gpu_A, gpu_b, gpu_x, gpu_oldx, err, order);
	check(cudaMemcpy(x, gpu_x, order * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(gpu_A);
	cudaFree(gpu_b);
	cudaFree(gpu_x);
	cudaFree(gpu_oldx);
	vector_print(x);
	return x;
}

