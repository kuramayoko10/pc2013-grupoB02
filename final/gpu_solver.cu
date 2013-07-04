#include <cuda.h>
#include <assert.h>
#include "extern.h"
#include "smatrix.h"

__global__ void kernel(float *a, float *b, float *c, unsigned order)
{
	int i = threadIdx.x + blockIdx.x;
	if (i<order)
		c[i] = a[i]+b[i];
}



float *solve(float *A, float *b)
{
	unsigned i;
	float *x;	
	float *gpu_x, *gpu_b, *gpu_c;
	x = vector_new(order);
	for (i=0; i<order; ++i)
		x[i] = 1;
	cudaMalloc((void **)&gpu_x, order * sizeof(float));
	cudaMalloc((void **)&gpu_b, order * sizeof(float));
	cudaMalloc((void **)&gpu_c, order * sizeof(float));
	assert(cudaMemcpy(gpu_x, x, order * sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
	assert(cudaMemcpy(gpu_b, b, order * sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
	kernel<<<order, 1>>>(gpu_x, gpu_b, gpu_c, order);	
	assert(cudaMemcpy(x, gpu_c, order * sizeof(float), cudaMemcpyDeviceToHost)==cudaSuccess);
	vector_print(x);
	cudaFree(gpu_c);
	cudaFree(gpu_b);
	cudaFree(gpu_x);
	return x;
}
