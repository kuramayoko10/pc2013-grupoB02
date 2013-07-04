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
}

