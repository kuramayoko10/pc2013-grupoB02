#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include "common.h"

#define NUMBER_OF_NODES 13

using namespace std;

//Contador de palindromos - palavra (wp) e frases (sp)
int wpCount = 0, spCount = 0;
//Tamanho do bloco de texto a ser repassado a cada no
int SIZE_OF_SEGMENT = (int)ceil(SMALL_TEXT_SIZE/NUMBER_OF_NODES);

/* Verifica se uma dada string e' palindromo
 * input: cadeia de caracteres a ser verificada
 *
 * return: true se a string for palindromo; false caso contrario.
 */
bool isPalindrome(const char *input);

/* Verifica se uma dado caractere e' um simbolo (nao e' letra)
 *
 * return: true se o caractere for simbolo; false caso seja uma letra.
 */
bool isSymbol(char input);

/* Verifica se um dado caractere representa o fim de uma sentenca
 * input: caractere a ser verificado
 *
 * return: true se o caractere for {.!?\n\r}; false caso contrario.
 */
bool endOfSentence(char input);

/* Retorna a proxima palavra dentro do arquivo apontado por fp
 * fp: ponteiro para um arquivo aberto no sistema
 * buffer: cadeia de caracteres que recebera o retorno da funcao
 *
 * return: 0 se nao ha mais nada a ser lido no arquivo, 1 se a palavra tiver mais de 3 letras, 2 caso a palavra tiver entre 0 e 2 letras.
 */
int readWordFromFile(FILE *fp, char *buffer);

/* Retorna a proxima sentenca dentro do arquivo apontado por fp. Ao processar a sentenca ja verifica se as palavras nela contidas sao palindromos
 * fp: ponteiro para um arquivo aberto no sistema
 * buffer: cadeia de caracteres que recebera o retorno da funcao
 * palindromes: ponteiro para o vector onde esta sendo armazenados os p lindromos ja classificados
 * primeList: ponteiro para o vector que contem todos os numeros primos de 2 a 20000
 *
 * return: 0 se nao ha mais nada a ser lido no arquivo, 1 se a sentenca tiver mais de 3 letras, 2 caso a sentenca tiver entre 0 e 2 letras.
 */

int readSentenceFromFile(FILE *fp, char *buffer, vector<Palindrome> *palindromes, vector<int> *primeList);

/* Adiciona a palavra word no vector de palindromos. Se a palavra ja tiver sido armazenada anteriormente, apenas incrementa um contador
 * palindromes: ponteiro para o vector onde esta sendo armazenados os p lindromos ja classificados
 * word: string que contem a palavra ou sentenca palindromo
 * primeList: ponteiro para o vector que contem todos os numeros primos de 2 a 20000
 * mode: caractere 'L' ou 'S', onde L habilitara a verificacao de numero primo em word
 *
 * return: a posicao no vector de palindromos no qual word esta armazenada.
 */
int addPalindrome(vector<Palindrome> *palindromes, string word, vector<int> *primeList, char mode);

/* Retorna a soma numerica dos caracteres ASCII que compoe a cadeia str
 * str: cadeia de caracteres a ser somada
 *
 * return: inteiro referente a soma numerica dos caracteres ASCII.
 */
int sumASCII(const char *str);

//Funcao principal
int main(int argc, char **argv){
	int numtasks, rank, rc, dest, source, tag=1;
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
        {
			SIZE_OF_SEGMENT = (int)ceil(5344213/NUMBER_OF_NODES);
        }
        
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
			/*aqui é feito a verificação se chegou a um final de frase, caso nao tenha chegado e já tenha sido o tamanho total que a partição 				aumenta, aumentamos o tamanho da partição e continuamos lendo ate encontrar um final de frase*/
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
		/*depois que dividimos todo o texto, enviamos cada partição para um nó, que ira fazer o processamento da sua particao*/
		for(dest = 1; dest <= NUMBER_OF_NODES; dest++){
			rc = MPI_Send(outmsg[dest-1], strlen(outmsg[dest-1]), MPI_CHAR, dest, tag, MPI_COMM_WORLD);
            free(outmsg[dest-1]);
		}
        fclose(file);
	}
	/*aqui são os nós que irao receber os textos e processalos*/
	else{
		source = 0;	
		int tamanho = 0;
		int ret = 2;
		char buffer[32768];
		if(!strcmp(argv[1], "wikipedia.txt"))
			tamanho = 54261766;
		else
			tamanho = 5344213;
		/*aloca-se o buffer com o tamanho do maior texto que pode ser lido, no caso, o texto inteiro. Logo em seguida o nó recebe a particao do 		texto via mensagem*/
		inmsg[rank-1] = (char*) malloc (tamanho*sizeof(char));
		rc = MPI_Recv(inmsg[rank-1], tamanho, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);

		//Escreve o texto que recebeu em um arquivo
		FILE *fileout;
		char vai[5];
        string resultado = "";
		sprintf(vai, "%i", rank);  
		fileout = fopen(vai, "w");
		fprintf(fileout, "%s\n", inmsg[rank-1]);	
		fclose(fileout);
		fileout = fopen(vai, "r");
	

        //Leia do arquivo recem criado cada palindromo e frase
        //Utiliza o metodo ja implementado no algoritmo sequencial
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
        free(inmsg[rank-1]);
        
        resultado.append(vai);
        resultado.append("-out.txt");
        fileout = fopen(resultado.c_str(), "w");
        /*depois de calcularmos as palindromes de cada partição, escrevemos os resultados encontrados em um arquivo com o nome da particao.out*/
        fprintf(fileout, "wp: %d / sp: %d\n", wpCount, spCount);
        for(int i = 0; i < palindromes.size(); i++){
            fprintf(fileout, "%s - %d occurrences", palindromes[i].word.c_str(), palindromes[i].count);
            if(palindromes[i].primeNumber != 0)
                fprintf(fileout, " - prime number %d", palindromes[i].primeNumber);
            fprintf(fileout, "\n");
        }
        
        fclose(fileout);
	}

	/*
	 * Finaliza a sessão MPI
	 */
	MPI_Finalize();
    
	return 0;
}


int readWordFromFile(FILE *fp, char *buffer)
{
    char read;
    int i = 0;
    
    buffer[0] = '\0';
    read = fgetc(fp);
    
    //Enquanto estiver lendo letras, compoe a palavra
    //Se encontrar simbolo, numero ou espaco em branco retorna a palavra formada
    while(!isSymbol(read) && !isspace(read))
    {
        buffer[i++] = read;
        read = fgetc(fp);
    }
    
    buffer[i] = '\0';
    
    //Consideramos palindromos apenas as palavras de mais de 3 caracteres.
    //Pois estas tem um significado claro na lingua
    //Artigos 'a' ou palavras comprimidas will -> "ll" nao nos interessa
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
    
    //Enquanto estiver lendo letras, compoe a sentenca
    //Se encontrar simbolo ".!?\n\r" ou fim de arquivo, retorna a sentenca formada
    while(!endOfSentence(read) && !feof(fp))
    {
        //Enquanto estiver lendo letras, compoe a palavra
        //Se encontrar simbolo, numero ou espaco em branco retorna a palavra formada
        if(isSymbol(read) || isspace(read))
        {
            //Descarta a pontuacao/espaco_branco
            word[w] = '\0';
            
            //Consideramos palindromos apenas as palavras de mais de 3 caracteres.
            if(w >= 3)
            {     
                if(isPalindrome(word))
                {
                    wpCount++;
                    addPalindrome(palindromes, word, primeList, 'S');
                }
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
    
    //Consideramos palindromos apenas as sentencas de mais de 3 caracteres.
    if(strlen(buffer) >= 3)
        return 1;
    
    if(feof(fp))
        return 0;
    
    return 2;
}

bool isSymbol(char input)
{
    //Caracteres aceitos sao: letras minisculas/maisculas
    if((input >= 65 && input <= 90) || (input >= 97 && input <= 122))
    {
        return false;
    }
    
    return true;
}

bool endOfSentence(char input)
{
    if(input == '.' || input == '!' || input == '?' || input == '\n' || input == '\r')
        return true;
    
    return false;
}

bool isPalindrome(const char *input)
{
    int size = (unsigned)strlen(input) - 1;
    int i = 0;
    
    //Percorre a palavra da esquerda->direita e direita->esquerda comparando as letras
    //Se alguma comparacao nao bater, a palavra nao eh palindromo
    //So precisa percorrer ate a metade do vetor
    while(i <= ceil(size/2.0))
    {
        if(size < 0)
            return false;
        
        //ignora diferenca de caixa-alta e caixa-baixa
        char a = toupper(input[i]);
        char b = toupper(input[size-i]);
        
        if(a != b)
            return false;
        
        i++;
    }
    
    return true;
}

int addPalindrome(vector<Palindrome> *palindromes, string word, vector<int> *primeList, char mode)
{
    int i = 0, ret = -1;
    unsigned long size = palindromes->size();
    clock_t startTimer, stopTimer;
    
    //Verifica se a palavra ja foi adicionada
    startTimer = clock();
    while(i < size && ret < 0)
    {
        ret = word.compare(palindromes->at(i).word);
        i++;
    }
    
    //Adiciona ao contador se a palavra ja estiver inserido
    if(ret == 0)
    {
        palindromes->at(i-1).count++;
    }
    else //Caso contrario insere a palavra no vetor na posicao que mantem a ordem alfabetica do mesmo
    {
        Palindrome pal;
        
        pal.word = word;
        pal.count = 1;
        pal.primeNumber = 0;
        
        if(i < 1)
            i = 1;
        
        palindromes->insert(palindromes->begin()+i-1, pal);
        
        //Se o modo de leitura eh Large, verifica que a palavra remete a um numero primo
        if(mode == 'L')
        {
            int num = sumASCII(word.c_str());
            
            clock_t primeStart = clock();
            if(isPrimeNumber(primeList, num))
            {
                palindromes->at(i-1).primeNumber = num;
                
                clock_t primeStop = clock();
                printf("Tempo Verificacao de Primo: %lf\n", (double)(primeStop - primeStart)/CLOCKS_PER_SEC);
            }
        }
    }
    
    stopTimer = clock();
    printf("Tempo Registro de Palindromo no Vector: %lf\n", (double)(stopTimer-startTimer)/CLOCKS_PER_SEC);
    
    return i-1;
}

int sumASCII(const char *str)
{
    int sum = 0;
    
    for(int i = 0; i < strlen(str); i++)
        sum += str[i];
    
    return sum;
}



