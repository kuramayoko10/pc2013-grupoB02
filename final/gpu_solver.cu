#include <cuda.h>
#include <assert.h>
#include <stdio.h>
#include "extern.h"
#include "smatrix.h"

//Numero de threads por bloco CUDA
#define THREADS_PER_BLOCK 250

//Metodo para capturar erro nos metodos de CUDA no host
//Imprime a mensagem de erro e a linha de codigo
#define check(X)                \
{                               \
	cudaError_t cerr = X;		\
	if (cerr != cudaSuccess){	\
		fprintf(stderr, "GPUassert:%s at line%d.\n", cudaGetErrorString(cerr), __LINE__);	\
		abort();                \
	}                           \
}

//Metodo para obter o valor absoluto de um float
#define my_abs(x) (x)>=0.0?(x):-1.0*(x)

//Metodo Jacobi-Richardson no Device
//Cada thread de um dado bloco resolve um x do vetor de incognitas
__global__ void kernel(float* A, float *b, float *x, float* oldx, float err, unsigned order) 
{ 
    //Obtem o id unico da thread em execucao no momento
	unsigned idx = threadIdx.x + blockDim.x * blockIdx.x; 
    
	unsigned i, it=0;
	float aux;
    
    //Se o id da thread estiver dentro do numero de linhas do vetor (ordem da matriz), executa o metodo
	if(idx < order)
	{ 
        //Variaveis para calculo do erro
        //Variaveis pertencem a memoria compartilhada do device - mais rapido
		__shared__ float maxxdiff; 
		__shared__ float maxx; 

		maxxdiff=1.0; 
		maxx=1.0; 

        //Enquanto o erro OU o numero de iteracoes nao atingir o limite esperado
        //continua realizando as iteracoes
		while((maxxdiff/maxx)>err || it < 150000/order)
		{ 
			x[idx] = 0.0; 
			maxxdiff = 0.0; 
			maxx=0.0; 
            
            //Percorre todas colunas da matriz, realizando multiplicacao pelo vetor 'x' anterior
            //Equivale a equacao (R*x)
			for (i = 0; i < order; ++i) 
			{ 
				x[idx]+=(i!=idx)?(smatrix_at(A, idx, i)*oldx[i]):0.0; 
			} 
            
            //Subtrai o resultado obtido acima do vetor 'b' e multiplica pela diagonal da matriz
            //Equacao: D*(b - R*x)
			x[idx] = (1.0/smatrix_at(A, idx, idx))*(b[idx]-x[idx]); 

            //Calcula o erro
			aux = my_abs(x[idx]);
			if (aux > maxx)
				maxx = aux;
			aux = abs(x[idx]-oldx[idx]);
			if (aux > maxxdiff)
				maxxdiff = aux;
                
            //Armazena em oldx os valores obtidos nesta iteracao
			oldx[idx]=x[idx]; 
            //Aumenta o contador de iteracoes
			it++;
		} 
	} 
}

//Solve - inicializa e aloca os vetores a serem usados no dispositivo
//Metodo chamado no main()
//
//@param A - matriz de coeficientes do sistema linear
//@param b - vetor com os valores `a direita da igualdade
//@return - o vetor com a solucao para todas as incognitas 'x'
float *solve(float *A, float *b)
{
	unsigned i;
	float *x;	
	float *gpu_A, *gpu_b, *gpu_x, *gpu_oldx;
	x = vector_new(order);
    
    //Inicializa x para 1
	for (i=0; i<order; ++i)
		x[i] = 1.0;
        
    //Aloca os vetores para a MatrizA, VetorB, VetorX (iteracao atual), VetorOldX (iteracao anterior)
	check(cudaMalloc((void **)&gpu_A, order * order * sizeof(float)));
	check(cudaMalloc((void **)&gpu_b, order * sizeof(float)));
	check(cudaMalloc((void **)&gpu_x, order * sizeof(float)));
	check(cudaMalloc((void **)&gpu_oldx, order * sizeof(float)));
    
    //Copia o conteudo destes vetores e sua referencia para o dispositivo
	check(cudaMemcpy(gpu_A, A, order * order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_b, b, order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_x, x, order * sizeof(float), cudaMemcpyHostToDevice));
	check(cudaMemcpy(gpu_oldx, x, order * sizeof(float), cudaMemcpyHostToDevice));
    
    //Chama o kernel que contem o metodo Jacobi implementado
	kernel<<<order/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gpu_A, gpu_b, gpu_x, gpu_oldx, err, order);
    
    //Recebe do dispositivo o vetor 'x' com a solucao obtida
	check(cudaMemcpy(x, gpu_x, order * sizeof(float), cudaMemcpyDeviceToHost));
    
    //Libera a memoria alocada na placa de video
	cudaFree(gpu_A);
	cudaFree(gpu_b);
	cudaFree(gpu_x);
	cudaFree(gpu_oldx);
    
	return x;
}

