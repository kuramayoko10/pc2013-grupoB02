#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "common.h"
#include "hmap.h"
#include "qrand.h"
#include "process.h"
#include "mpi.h"

/* Arranjos para armazenar palavras, em ordem alfabetica, com
 * less: Cinco ou menos letras
 * more: Mais de cinco letras
 * all: todas palavras fornecidas*/
char *less_array[N_LESS_WORD], *more_array[N_MORE_WORD], *all_array[N_ALL_WORD];
// Hash Map para palavras de cinco ou menos letras
struct hmap *less_map;
// Contadores de numeros de clock
// Nao funcionam bem com OMP. Os tempos finais foram obtidos pela funcao time() do sistema operacional
clock_t start, finish;

/* Carrega as estruturas iniciais na memoria (arranjos)
 * Sem parametros e sem retorno
 */
void init();
/* Libera memoria das estruturas alocadas em init()
 * Sem parametros e sem retorno
 */
void end();
/* Funcao que gera aleatoriamente todas palavras pequenas (Cinco ou menos letras)
 * Sem parametros e sem retorno
 */
void process_less();
/* Funcao que gera por composicao (concatenacao) palavras grandes
 * @param argc - numero de argumentos passados na execucao do programa (utilizado pelo MPI)
 * @param argv - lista de argumentos passados na execucao do programa (utilizado pelo MPI)
 * Sem retorno
 */
void process_more(int argc, char **argv);
/* Dado uma palavra grande, retorna se foi possivel compo-la a partir de palavras menores
 * @param array - arranjo que contem todas palavras menores
 * @param tam_array - tamanho do arranjo array (numero de palavras)
 * @param finalWord - palavra que deseja-se comport
 * @return - 1 caso foi possivel realizar a composicao. 0 caso contrario
 */
int word_compound(char **array, int tam_array, char *finalWord);
/* Dado uma palavra grande e uma cadeia de caracteres, retorna se a cadeia menor e' substring (faz parte) da palavra maior
 * @param sub_string - cadeia de caracteres pequena
 * @param final_word - palavra grande
 * @return - 0 caso sub_string seja igual a final_word. -1 caso sub_string nao faca parte de final_word. Tamanho da sub_string
 *              caso sub_string faca parte de final_word
 */
int is_substring(char *sub_string, char *final_word);

// Funcao principal
int main(int argc, char **argv)
{
    //Inicia uma semente para a funcao rand() da libc
	qrand_seed((unsigned)time(NULL));
    
	init();
    
	start = clock();
	process_less();
    finish = clock();
    printf("process_less demorou: %fs.\n", (float)(finish-start)/CLOCKS_PER_SEC);
    
    start = clock();
	process_more(argc, argv);
	finish = clock();
	printf("Process_more demorou: %fs.\n", (float)(finish-start)/CLOCKS_PER_SEC);
    
	end();
    
	return SUCCESS;
}

/* Calcula o valor da chave hash para uma string
 * Utiliza alguns numeros magicos.
 */
unsigned long string_hash(const void *a)
{
	const unsigned char *str = a;
	unsigned long hash = 5381;
	int c;
	while ((c = *str++))
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	return (hash)%MAP_SIZE;
}

/* Carrega as estruturas iniciais na memoria (arranjos) */
void init(void)
{
	unsigned i;                                                        
	bool flag_found=FALSE;                                             
	char input_string[40];
	FILE *less_file, *more_file;
	less_map = hmap_init(MAP_SIZE, 6*sizeof(char), sizeof(bool), string_hash); 
	less_file = fopen("less.txt", "r");                                
	more_file = fopen("more.txt", "r");                                

	for (i=0, memset(input_string, '\0', 6); fscanf(less_file, "%s", input_string) != EOF; i++) 
	{                                                                  
		less_array[i] = (char*)malloc(sizeof(char)*6);             
		strcpy(less_array[i], input_string);                       
		hmap_insert(less_map, less_array[i], &flag_found);        
		memset(input_string, 0, 6);
	}
    
	for (i=0; fscanf(more_file, "%s", input_string) != EOF; i++)   
	{
		more_array[i] = (char*)malloc(sizeof(char)*41);            
		strcpy(more_array[i], input_string);                       
	}
    
	fclose(less_file);                                                 
	fclose(more_file);                                                 
}       

/* Libera memoria das estruturas alocadas em init() */
void end(void)
{
	unsigned i;
	hmap_free(less_map);                                              
	for(i = 0; i < N_LESS_WORD; i++)                                  
		free(less_array[i]);                                   
	for(i = 0; i < N_MORE_WORD; i++)                               
		free(more_array[i]);
}

/* Funcao que gera aleatoriamente todas palavras pequenas (Cinco ou menos letras) */
void process_less(void)
{
	unsigned i;
	char rand_str[6];
	bool found_flag;
    
    //Divide a geracao de palavras pequenas entre cada nucleo
    //Variaveis privadas a cada nucleo sao: a string sendo gerada e a flag que marca se a palavra existe no hash 
#pragma omp parallel private(rand_str, found_flag)
	for (i = 0; i < N_LESS_WORD;)
	{
        //Gera a palavra randomicamente
		qrand_word(rand_str);
        
        //Verifica se a palavra se encontra no hash
		if (hmap_search(less_map, rand_str, &found_flag)==SUCCESS)
#pragma omp critical
		{
			hmap_remove(less_map, rand_str, &found_flag);
			//printf("Found:%s\n", rand_str);
			i++;
		}
	}
}

/* Dado uma palavra grande, retorna se foi possivel compo-la a partir de palavras menores */
void process_more(int argc, char **argv)
{
	int rank, dest, source, tag=1, i=0;
	MPI_Status Stat;
	int total=0;
	char *smallWordArray[N_LESS_WORD], input_string[40];
    FILE *fLess;
	fLess = fopen("less.txt", "r");

    //copia as palavras com tamanho menor que 6 para um array, pois fica mais rapido o processamento
    while(fscanf(fLess, "%s", input_string) != EOF)
    {
        smallWordArray[i] = (char*)malloc(sizeof(char)*6);
        strcpy(smallWordArray[i], input_string);
        i++;
    }

	//Inicia uma sessão MPI
	MPI_Init(&argc, &argv);
    
	//Obtêm o id do processo
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    //Atividades do no mestre
	if (rank == 0)
    {
		int posicao = 0;
        
		//E' enviado a posicao que cada no escravo deve ler dentro do arquivo
		for(dest = 1; dest <= NUMBER_OF_NODES; dest++)
        {
			rc = MPI_Send(&posicao, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
			posicao += SIZE_TO_READ; 
		}
	}
	else //Atividades dos nos escravos
    {
		source = 0;	
		int posicao = 0, count=0;
		char *compoundWordArray[SIZE_TO_READ];
		FILE *fMore;
		fMore = fopen("more.txt", "r");
        
        //Recebe qual posicao ele deve iniciar lendo do arquivo
		MPI_Recv(&posicao, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &Stat);
        
		fseek(fMore, posicao, SEEK_SET);

		//Posiciona para cada no o local certo do arquivo para este ler essa parte apenas
		for(;count < SIZE_TO_READ; count++)
        {
	        compoundWordArray[count] = (char*)malloc(sizeof(char)*41);
			fscanf(fMore, "%s", compoundWordArray[count]);
		}
		fclose(fMore);

		//Contagem de palavras compostas
        //As palavras sao divididas entre os 4 nucleos da maquina usando OMP
#pragma omp parallel for
	    for(count = 0; count < SIZE_TO_READ; count++){
	        if(word_compound(smallWordArray, N_LESS_WORD, compoundWordArray[count])){
	        	//printf("Found: %s\n", compoundWordArray[count]);
#pragma omp critical
            	total++;
        	}
    	}
	}

	//Finaliza a sessão MPI
	fclose(fLess);
	MPI_Finalize();
    
    //Imprime resultado final
    printf("Rank %d - Total: %d\n", rank, total);
}

/* Para fazer a composicao de palavras utilizamos a seguiten lógica:
 * Buscamos dentro do array de palavras menores que 6 letras por uma palavra 
 * que comece com a mesma letra da palavra grande buscada.
 * Caso encontremos um pedaco da palavra (prefixo), chamamos recursivamente a funcao, para o restante da palavra
*/
int word_compound(char **array, int tam_array, char *finalWord)
{
	int i, ret;
	char buffer[41];

	for(i = 0; i < tam_array; i++)
	{
		int temp = is_substring(array[i], finalWord);
		int j;


		if(temp == 0)
			return 1;

		if(temp == -1)
		{
			continue;
		}
		else
		{
			int aux = abs((int)strlen(finalWord) - temp);
			//copia o restante da palavra maior para um buffer, logo em seguida a funcao eh chamada recursivamente para o que sobrou da palavra 			maior, no caso o buffer
			for(j = 0; j < aux; j++)
			{
				buffer[j] = finalWord[temp+j];
			}
			buffer[j] = '\0';

			ret = word_compound(array, tam_array, buffer);

			if(ret == 1)
				return 1;
		}

	}

	return 0;
}

/* Funcao destinada para verificar se as letras de uma string pertencem a uma outra
 * No caso para ver se uma palavra com tamanho menor que 6 letras pertence a uma com tamanho maior
 */
int is_substring(char *sub_string, char *final_word)
{
	int i = 0, tam = (int)strlen(sub_string);

	if((int)strlen(final_word) < tam)
		return -1;

	if(!strcmp(sub_string, final_word))
		return 0;
	for(;i<tam;i++)
		if(sub_string[i] != final_word[i])
			return -1;

	return (int)strlen(sub_string);
}

