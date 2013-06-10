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


char *less_array[N_LESS_WORD], *more_array[N_MORE_WORD], *all_array[N_ALL_WORD];
struct hmap *less_map;
clock_t start, finish;

void init();
void end();
void process_less();
void process_more(int argc, char **argv);
int word_compound(char **array, int tam_array, char *finalWord);
int word_compound_aux(char **array, int pos, char *final_word);
int is_substring(char *sub_string, char *final_word);


int main(int argc, char **argv)
{
	qrand_seed((unsigned)time(NULL));
	init();
	start=clock();
	//process_less();
	process_more(argc, argv);
	finish = clock();
	printf("Took %fs.\n", (float)(finish-start)/CLOCKS_PER_SEC);
	end();
	return SUCCESS;
}

/* 	Calculates the hash of a null terminated string by using some magic
 * numbers.
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

void init(void)
{
	unsigned i;                                                        
	bool flag_found=FALSE;                                             
	char input_string[40];
	FILE *less_file, *more_file;
	less_map = hmap_init(MAP_SIZE, 6*sizeof(char), sizeof(bool), string_hash); 
	less_file = fopen("less.txt", "r");                                
	more_file = fopen("more.txt", "r");                                

	//less_array = (char*)malloc(sizeof(char)*N_LESS_WORD);
	//more_array = (char*)malloc(sizeof(char)*N_MORE_WORD);
	//all_array = (char*)mallloc(sizeof(char)*N_ALL_WORD);

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

void end(void)
{
	unsigned i;
	hmap_free(less_map);                                              
	for(i = 0; i < N_LESS_WORD; i++)                                  
		free(less_array[i]);                                   
	for(i = 0; i < N_MORE_WORD; i++)                               
		free(more_array[i]);
}


void process_less(void)
{
	unsigned i;
	char rand_str[6];
	bool found_flag;
#pragma omp parallel private(rand_str, found_flag)
	for (i=0; i<N_LESS_WORD;)
	{
		qrand_word(rand_str);
		if (hmap_search(less_map, rand_str, &found_flag)==SUCCESS)
#pragma omp critical
		{
			hmap_remove(less_map, rand_str, &found_flag);
			//printf("Found:%s\n", rand_str);
			i++;
		}
	}
}

void process_more(int argc, char **argv)
{
	int rank, dest, source, tag=1, i=0;
	MPI_Status Stat;
	int total=0;
	char *smallWordArray[N_LESS_WORD], input_string[40];
    FILE *fLess;
	fLess = fopen("less.txt", "r");

//copia as palavras com tamanho menor que 6 para um array, pois fica mais rapido o processamento
    while(fscanf(fLess, "%s", input_string) != EOF){
        smallWordArray[i] = (char*)malloc(sizeof(char)*6);
        strcpy(smallWordArray[i], input_string);
        i++;
    }

	/*
	 * Inicia uma sessão MPI
	 */
	MPI_Init(&argc, &argv);
    
	/*
	 * Obtêm o id do processo
	 */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0){
		int posicao = 0;
		//eh enviado a posicao que cada no escravo deve ler dentro do arquivo
		for(dest = 1; dest <= NUMBER_OF_NODES; dest++){
			rc = MPI_Send(&posicao, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
			posicao += SIZE_TO_READ; 
		}
	}
	else{
		source = 0;	
		int posicao = 0, count=0;
		char *compoundWordArray[SIZE_TO_READ];
		FILE *fMore;
		fMore = fopen("more.txt", "r");
		MPI_Recv(&posicao, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &Stat);
		fseek(fMore, posicao, SEEK_SET);
		/*posiciona para cada no o local certo do arquivo para este ler essa parte apenas*/
		for(;count < SIZE_TO_READ; count++){
	        compoundWordArray[count] = (char*)malloc(sizeof(char)*41);
			fscanf(fMore, "%s", compoundWordArray[count]);
		}
		fclose(fMore);

		/*Contagem de palavras compostas*/
#pragma omp parallel for
	    for(count = 0; count < SIZE_TO_READ; count++){
	        if(word_compound(smallWordArray, N_LESS_WORD, compoundWordArray[count])){
	        	//printf("Found: %s\n", compoundWordArray[count]);
#pragma omp critical
            	total++;
        	}
    	}
	}

	/*
	 * Finaliza a sessão MPI
	 */
	fclose(fLess);
	MPI_Finalize();
    	printf("Rank %d - Total: %d\n", rank, total);
}

/*
**	para fazer a composicao de palavras utilizamos a seguiten lógica: Buscamos dentro do array de palavras menores que 6 letras por uma palavra **	que comece uma das palavras das palavras com tamanho maior que 5. Caso encontremos tal palavra, chamamos recursivamente a funcao, para o resto **	da palavra(as que sao maiores que 5) que sobrou, caso consiga completar a palavra, eh encontrada. 
**		Os parametros de entrada sao o array de palavras com tamanho menor que 6, o tamanho desse array, e uma palavra com tamanho maior que 5.
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

/*funcao destinada para verificar se as letras de uma string pertencem a outra(no caso para ver se uma palavra com tamanho menor que 6 pertence a uma com tamanho maior que 5*/
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

