#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>


#define SIZE_OF_SEGMENT 3875000
#define NUMBER_OF_NODES 14

int main(int argc, char **argv)
{
	int numtasks, rank, rc, dest, source, tag=1;
	char *outmsg[NUMBER_OF_NODES]; 
	char *inmsg;
	char fileName[30], aux;
	register unsigned long int count=0, count2=0, incrementOfSize=1;
	unsigned long int sizeOfVector[NUMBER_OF_NODES], maior = SIZE_OF_SEGMENT;
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
			if(count > maior)
				maior = count;
		}
		/*assim eu garanto que toda msg que eu receber poderá ser armazenada no vetor inmsg*/
		inmsg = (char*) malloc (maior*sizeof(char));
		
		for(dest = 1; dest <= NUMBER_OF_NODES; dest++)
			rc = MPI_Send(&outmsg[dest-1], sizeOfVector[dest-1], MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}
	else{
		source = 0;	
		printf("rank:%d\n", rank);	
		rc = MPI_Recv(&inmsg, sizeOfVector[rank-1], MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
		/*aqui chama a funcao sequencial para calcular se eh palindrome ou nao*/
	}

	/*
	 * Finaliza a sessão MPI
	 */
	MPI_Finalize();
	return 0;
}

