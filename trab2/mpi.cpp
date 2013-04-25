#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUMBER_OF_NODES 13
#define SIZE_OF_SEGMENT (int)ceil(54261766/NUMBER_OF_NODES)


int main(int argc, char **argv)
{
	int numtasks, rank, rc, dest, source, tag=1;
	char *outmsg[NUMBER_OF_NODES]; 
	char *inmsg[NUMBER_OF_NODES];
	char fileName[30], aux;
	register unsigned long int count=0, count2=0, incrementOfSize=1;
	unsigned long int sizeOfVector[NUMBER_OF_NODES];
	FILE *file;
	MPI_Status Stat;


	//printf("Digite o nome do arquivo de entrada.\n");
	//scanf("%s", fileName);
	//file = fopen(fileName, "r");

	/*
	 * Inicia uma sessão MPI
	 */
	MPI_Init(&argc, &argv);
	/*
	 * Obtêm o id do processo
	 */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0){
		file = stdin;
		if(!file){
	     	printf("Arquivo de entrada nao encontrado!\n");
	        exit(1);
	    }
		/*inicializa os vetores de caracteres para enviar a msg*/
		for(count=0; count < NUMBER_OF_NODES; count++)
			outmsg[count] = (char*) malloc (SIZE_OF_SEGMENT*sizeof(char ));
			
		for(count=0; count < NUMBER_OF_NODES; count++)
			outmsg[count][0] = '\0';
		
		/*para cada vetor, completa com trechos do texto, chegando ate o seu máximo, depois disso, continua a inserir dados no vetor até que seja 			encontrado um final de frase(no caso ponto, ponto de exclamação, interrogação ou ENTER)*/
		for(count2=0; count2 < NUMBER_OF_NODES; count2++){
			for(count=0;!feof(file) && count < SIZE_OF_SEGMENT; count++){
				fscanf(file, "%c", &outmsg[count2][count]);
			}
			if(!feof(file))
				fscanf(file, "%c", &aux);
			while(!feof(file) && aux != '.' && aux != '!' && aux != '?' && aux != '\n'){
				if(count == SIZE_OF_SEGMENT+1000*(incrementOfSize-1)){
		  			outmsg[count2] = (char *) realloc (outmsg[count2], (SIZE_OF_SEGMENT * sizeof(char)+ 1000*incrementOfSize));
	  				incrementOfSize++;		
				}			
				outmsg[count2][count] = aux;
				fscanf(file, "%c", &aux);
				count++;
			}
			incrementOfSize = 1;
			outmsg[count2][count] = '\0';
			sizeOfVector[count2] = count; 
		}
		for(dest = 1; dest <= NUMBER_OF_NODES; dest++){
			rc = MPI_Send(&outmsg[dest-1], sizeOfVector[dest-1], MPI_CHAR, dest, tag, MPI_COMM_WORLD);
		}
	}
	else{
		source = 0;	
		printf("rank:%d\n", rank);	
		int tamanho;
		MPI_Get_count(&Stat,MPI_CHAR,&tamanho);

		/*assim eu garanto que toda msg que eu receber poderá ser armazenada no vetor inmsg*/
		inmsg[rank-1] = (char*) malloc (tamanho*sizeof(char));
//		printf("%d\n", tamanho);
				
		rc = MPI_Recv(&inmsg[rank-1], tamanho, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
	
		

		/*aqui chama a funcao sequencial para calcular se eh palindrome ou nao*/
	}

	/*
	 * Finaliza a sessão MPI
	 */
	MPI_Finalize();
	return 0;
}

