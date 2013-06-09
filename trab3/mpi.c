
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "process.h"
#include "mpi.h"

#define NUMBER_OF_NODES 13
#define SIZE_TO_READ (int)ceil(N_COMPOUND_WORD/NUMBER_OF_NODES)

int wordCompound(char **array, int threshold, char *finalWord);

//Funcao principal
int main(int argc, char **argv){
	int numtasks, rank, rc, dest, source, tag=1, i=0;
	MPI_Status Stat;
	int total=0;
	char *smallWordArray[N_SMALL_WORD], input_string[40];
    FILE *fLess;
	fLess = fopen("less.txt", "r");

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
        printf("rank: %d\n", rank);
		int posicao = 0;
		/*depois que dividimos todo o texto, enviamos cada partição para um nó, que ira fazer o processamento da sua particao*/
		for(dest = 1; dest <= NUMBER_OF_NODES; dest++){
			rc = MPI_Send(&posicao, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
			posicao += SIZE_TO_READ; 
		}
	}
	/*aqui são os nós que irao receber os textos e processalos*/
	else{
        printf("rank: %d\n", rank);
        
		source = 0;	
		int posicao = 0, count=0, rec;
		char *compoundWordArray[SIZE_TO_READ], teste[40];
		FILE *fMore;
		fMore = fopen("more.txt", "r");
		rc = MPI_Recv(&posicao, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &Stat);
		fseek(fMore, posicao, SEEK_SET);
		for(;count < SIZE_TO_READ; count++){
	        compoundWordArray[count] = (char*)malloc(sizeof(char)*41);
			fscanf(fMore, "%s", compoundWordArray[count]);
		}
		fclose(fMore);
	
		/*Contagem de palavras compostas*/
	    for(count = 0; count < SIZE_TO_READ; count++){
	        if(wordCompound(smallWordArray, N_SMALL_WORD, compoundWordArray[count])){
	        	printf("Found: %s\n", compoundWordArray[count]);
            	total++;
        	}
    	}


		
	}

	/*
	 * Finaliza a sessão MPI
	 */
	fclose(fLess);
	MPI_Finalize();
    printf("Total: %d\n", total);
	return 0;
}




int wordCompound(char **array, int threshold, char *finalWord)
{
    int i, j;
    char *concat = (char*)malloc(sizeof(char)*40);
    
    for(i = 0; i < threshold; i++)
    {
        if(array[i][0] == finalWord[0])
        {
            for(j = 0; j < threshold; j++)
            {
                strcpy(concat, array[i]);
                strcat(concat, array[j]);
                
                if(strcmp(concat, finalWord) == 0)
                {
  //                  printf("Found: %s\n", finalWord);
                    return 1;
                }
            }
        }
    }
    
    return 0;
}


