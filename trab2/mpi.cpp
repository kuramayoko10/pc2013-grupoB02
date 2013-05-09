#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <string>
#include <vector>
#include "common.h"

#define NUMBER_OF_NODES 13
#define STATE_ALNUM 0
#define STATE_SPACE 1
#define STATE_PUNCT 2

using namespace std;

int wpCount = 0, spCount = 0;

bool isPalindrome(const char *input);
bool isSymbol(char input);
bool endOfSentence(char input);
int readWordFromFile(FILE *fp, char *buffer);
int readSentenceFromFile(FILE *fp, char *buffer, vector<Palindrome> *palindromes, vector<int> *primeList);
int addPalindrome(vector<Palindrome> *palindromes, string word, vector<int> *primeList, char mode);
int sumASCII(const char *str);



//int addPalindrome(vector<Palindrome> *palindromes, char *word, int wordSize);
//void readAndProcessBuffer(char *text, int size, vector<Palindrome> *palindromes);

int main(int argc, char **argv){
	int numtasks, rank, rc, dest, source, tag=1, SIZE_OF_SEGMENT;
	char *outmsg[NUMBER_OF_NODES]; 
	char *inmsg[NUMBER_OF_NODES];
	char fileName[30], aux;
	register unsigned long int count=0, count2=0, incrementOfSize=1;
	FILE *file;
	MPI_Status Stat;
    vector<Palindrome> palindromes;
	vector<int> primeList;
	sievePrimeNumbers(&primeList, 20000);
	/*
	 * Inicia uma sessão MPI
	 */
	MPI_Init(&argc, &argv);
	/*
	 * Obtêm o id do processo
	 */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0){
		file = fopen(argv[1], "r");
		if(!strcmp(argv[1], "wikipedia.txt"))
			SIZE_OF_SEGMENT = (int)ceil(54261766/NUMBER_OF_NODES);
		else
			SIZE_OF_SEGMENT = (int)ceil(5344213/NUMBER_OF_NODES);
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
				outmsg[count2][count] = tolower(outmsg[count2][count]);
			}
			if(!feof(file))
				fscanf(file, "%c", &aux);
			while(!feof(file) && aux != '.' && aux != '!' && aux != '?' && aux != '\n'){
				if(count == SIZE_OF_SEGMENT+1000*(incrementOfSize-1)){
		  			outmsg[count2] = (char *) realloc (outmsg[count2], (SIZE_OF_SEGMENT * sizeof(char)+ 1000*incrementOfSize));
	  				incrementOfSize++;		
				}			
				outmsg[count2][count] = aux;
				outmsg[count2][count] = tolower(outmsg[count2][count]);
				fscanf(file, "%c", &aux);
				count++;
			}
			incrementOfSize = 1;
		}

		for(dest = 1; dest <= NUMBER_OF_NODES; dest++){
			rc = MPI_Send(outmsg[dest-1], strlen(outmsg[dest-1]), MPI_CHAR, dest, tag, MPI_COMM_WORLD);
		
		}
	}
	else{
		source = 0;	
		printf("rank:%d\n", rank);
		int tamanho = 0;
		int ret = 2;
		char buffer[2048];
		if(!strcmp(argv[1], "wikipedia.txt"))
			tamanho = 54261766;
		else
			tamanho = 5344213;
		inmsg[rank-1] = (char*) malloc (tamanho*sizeof(char));

		rc = MPI_Recv(inmsg[rank-1], tamanho, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);

		//escreve o texto dividido em arquivos
		FILE *fileout;
		char vai[5];
		sprintf(vai, "%i", rank);  
		fileout = fopen(vai, "w");
		fprintf(fileout, "%s\n", inmsg[rank-1]);	
		fclose(fileout);
		fileout = fopen(vai, "r");
	
		while(ret){
			if(!strcmp(argv[1], "wikipedia.txt")){
				ret = readWordFromFile(fileout, buffer);
				 if(ret == 1){
                   if(isPalindrome(buffer)){
                   //Adiciona o palindromo ao vector de palindromos
                      addPalindrome(&palindromes, buffer, &primeList, 'L');
                      wpCount++;
                   }
                }
			}
			else{
				ret = readSentenceFromFile(fileout, buffer, &palindromes, &primeList);
                if(ret == 1){
                   if(isPalindrome(buffer)){
                       addPalindrome(&palindromes, buffer, &primeList, 'S');
                       spCount++;
                   }
                }
			}
		}
		fclose(fileout);
		
	}

	/*
	 * Finaliza a sessão MPI
	 */
	MPI_Finalize();
		
	for(int i = 0; i < palindromes.size(); i++){
        printf("%s - %d occurrences", palindromes[i].word.c_str(), palindromes[i].count);
        
        if(palindromes[i].primeNumber != 0)
            printf(" - prime number %d", palindromes[i].primeNumber);
        
        printf("\n");
    }
	fclose(file);
	for(count = 0; count < NUMBER_OF_NODES; count++){
		free(inmsg[count]);
		free(outmsg[count]);
	}
	return 0;
}


int readWordFromFile(FILE *fp, char *buffer)
{
    char read;
    int i = 0;
    
    buffer[0] = '\0';
    read = fgetc(fp);
    
    while(!isSymbol(read) && !isspace(read))
    {
        buffer[i++] = read;
        read = fgetc(fp);
    }
    
    buffer[i] = '\0';
    
    //Consideramos palindromos apenas as palavras de mais de 3 caracteres.
    //Pois estas tem um significado claro na lingua
    //Artigos 'a' ou palavras comprimidas "we'll" nao nos interessa
    if(strlen(buffer) > 2)
    {
        return 1;
    }
    
    if(feof(fp))
        return 0;
    
    return 2;
}

int readSentenceFromFile(FILE *fp, char *buffer, vector<Palindrome> *palindromes, vector<int> *primeList)
{
    char read;
    char word[128];
    int i = 0;
    int w = 0;
    
    buffer[0] = '\0';
    read = fgetc(fp);
    
    while(!endOfSentence(read) && !feof(fp))
    {
        if(isSymbol(read) || isspace(read))
        {
            //Processa a palavra anterior e descarta a pontuacao/espaco_branco
            word[w] = '\0';
            
            if(w >= 3)
            {
                //printf("%d - %s\n", i, word);
                
                if(isPalindrome(word))
                {
                    wpCount++;
                    addPalindrome(palindromes, word, primeList, 'S');
                }
                
                //printf("%s\n", word);
            }
            
            w = 0;
        }
        else
        {
            buffer[i++] = read;
            word[w++] = read;
        }
        
        read = fgetc(fp);
    }
    buffer[i] = '\0';
    
    //Consideramos palindromos apenas as palavras de mais de 3 caracteres.
    //Pois estas tem um significado claro na lingua
    //Artigos 'a' ou palavras comprimidas "we'll" nao nos interessa
    if(strlen(buffer) >= 3)
        return 1;
    
    if(feof(fp))
        return 0;
    
    return 2;
}

bool isSymbol(char input)
{
    //Caracteres aceitos sao: letras minisculas/maisculas
    //(input >= 48 && input <= 57) || 
    if((input >= 65 && input <= 90) || (input >= 97 && input <= 122))
    {
        return false;
    }
    
    return true;
}

bool endOfSentence(char input)
{
    if(input == '.' || input == '!' || input == '?' || input == '\n')
        return true;
    
    return false;
}

bool isPalindrome(const char *input)
{
    int size = (unsigned)strlen(input) - 1;
    int i = 0;
    
    while(i <= ceil(size/2.0))
    {
        if(size < 0)
            return false;
        
        //Percorre a palavra da esquerda->direita e direita->esquerda comparando as letras
        //Se alguma comparacao nao bater, a palavra nao eh palindromo
        
        //ignora caps
        char a = toupper(input[i]);
        char b = toupper(input[size-i]);
        
        if(a != b)
            return false;
        
        i++;
    }
    
    return true;
}

//Adiciona a palavra oa vetor e retorna sua posicao
int addPalindrome(vector<Palindrome> *palindromes, string word, vector<int> *primeList, char mode)
{
    int i = 0, ret = -1;
    unsigned long size = palindromes->size();
    
    //Verifica se a palavra ja foi adicionada
    while(i < size && ret < 0)
    {
        ret = word.compare(palindromes->at(i).word);
        i++;
    }
    
    //Adiciona o contador
    if(ret == 0)
    {
        palindromes->at(i-1).count++;
    }
    else
    {
        Palindrome pal;
        
        pal.word = word;
        pal.count = 1;
        pal.primeNumber = 0;
        
        if(i < 1)
            i = 1;
        
        palindromes->insert(palindromes->begin()+i-1, pal);
        
        if(mode == 'L')
        {
            int num = sumASCII(word.c_str());
            
            if(isPrimeNumber(primeList, num))
            {
                palindromes->at(i-1).primeNumber = num;
            }
        }
    }
    
    return i-1;
}

int sumASCII(const char *str)
{
    int sum = 0;
    
    for(int i = 0; i < strlen(str); i++)
        sum += str[i];
    
    return sum;
}





