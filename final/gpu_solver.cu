#include <cuda.h>
#include <assert.h>
#include "extern.h"
#include "smatrix.h"

#define check(X) assert(X==cudaSuccess)

__global__ void kernel(float *a, float *b, float *c, unsigned order)
{
	int i = threadIdx.x + blockIdx.x;
	if (i<order)
		c[i] = a[i]+b[i];
}



float *solve(float *A, float *b)
{
	unsigned i, it=0;
	float *x;	
	float *gpu_A, *gpu_x, *gpu_b, *gpu_c;
	x = vector_new(order);
	for (i=0; i<order; ++i)
		x[i] = 1.0;
	check(cudaMalloc((void **)&gpu_A, order * order *sizeof(float)));
	check(cudaMalloc((void **)&gpu_b, order * sizeof(float)));
	check(cudaMemcpy(gpu_A, A, order * order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_b, b, order * sizeof(float), cudaMemcpyHostToDevice));
	while (it<it_num)
	{
		cudaMemcpy(gpu_x, x, order * sizeof(float), cudaMemcpyHostToDevice);
		kernel<<<order, 1>>>(gpu_x, gpu_b, gpu_c, order);	
		cudaMemcpy(x, gpu_x, order * sizeof(float), cudaMemcpyDeviceToHost);
		++it;
	}
	cudaFree(gpu_A);
	cudaFree(gpu_b);
	vector_print(x);
	return x;
}

