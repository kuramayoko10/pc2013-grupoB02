// sequencial.cpp - Trabalho 2
// Deteccao de palindromos em textos
// Algoritmo sequencial e sem paralelizacao
//
// Grupo: 02
// Cassiano K. Casagrande, Guilherme S. Gibertoni, Rodrigo V. C. Beber

#include "common.h"

using namespace std;

/* Verifica se uma dada string e' palindromo
 * input: cadeia de caracteres a ser verificada
 * callCount: inteiro para armazenar a contagem de chamadas a este metodo
 *
 * return: true se a string for palindromo; false caso contrario.
 */
bool isPalindrome(const char *input, int *callCount);

/* Verifica se uma dado caractere e' um simbolo (nao e' letra)
 * input: caractere a ser verificado
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
 * isPalCount: inteiro para armazenar a contagem de chamadas ao metodo isPalindrome()
 * addPalCount: inteiro para armazenar a contagem de chamadas ao metodo addPalindrome()
 * isPrimeCount: inteiro para armazenar a contagem de chamadas ao metodo isPrimeNumber()
 *
 * return: 0 se nao ha mais nada a ser lido no arquivo, 1 se a sentenca tiver mais de 3 letras, 2 caso a sentenca tiver entre 0 e 2 letras.
 */

int readSentenceFromFile(FILE *fp, char *buffer, vector<Palindrome> *palindromes, vector<int> *primeList, int *isPalCount, int *addPalCount, int *isPrimeCount);

/* Adiciona a palavra word no vector de palindromos. Se a palavra ja tiver sido armazenada anteriormente, apenas incrementa um contador
 * palindromes: ponteiro para o vector onde esta sendo armazenados os p lindromos ja classificados
 * word: string que contem a palavra ou sentenca palindromo
 * primeList: ponteiro para o vector que contem todos os numeros primos de 2 a 20000
 * mode: caractere 'L' ou 'S', onde L habilitara a verificacao de numero primo em word
 * callCount: inteiro para armazenar a contagem de chamadas a este metodo
 * isPrimeCount: inteiro para armazenar a contagem de chamadas ao metodo isPrimeNumber()
 *
 * return: a posicao no vector de palindromos no qual word esta armazenada.
 */
int addPalindrome(vector<Palindrome> *palindromes, string word, vector<int> *primeList, char mode, int *callCount, int *isPrimeCount);

/* Retorna a soma numerica dos caracteres ASCII que compoe a cadeia str
 * str: cadeia de caracteres a ser somada
 *
 * return: inteiro referente a soma numerica dos caracteres ASCII.
 */
int sumASCII(const char *str);

//Contadores de numeros de palavras palindromos (wp) e sentencas palindromos (sp)
int wpCount = 0;
int spCount = 0;


//Funcao principal
int main(int argc, const char * argv[])
{
    FILE *inputFile;
    vector<int> primeList;
    vector<Palindrome> palindromes;
    
    char word[1024];        //Armazena a palavra encontrada. Max de 1023 letras
    char sentence[10240];   //Armazena a sentenca encontrada. Max de 10239 letras
    char mode = 'a';
    char *fileBuffer = '\0';
    
    clock_t startTimer, stopTimer;
    int ret = 1;
    int isPalCount = 0, addPalCount = 0, isPrimeCount = 0;
    
    //A execucao so sera iniciada se 2 argumentos adicionais forem providenciados: nome do arquivo de entrada e modo de leitura
    if(argc >= 3)
    {
        //Carrega o arquivo passado no segundo argumento
        inputFile = fopen(argv[1], "r");
        
        if(!inputFile)
        {
            printf("File not found!\n");
            exit(1);
        }
        
        //Carrega o modo de leitura, passado no terceiro argumento
        if(strlen(argv[2]) == 1)
            mode = argv[2][0];
        
        //Inicia o processamento
        //A leitura do texto eh realizada a partir do arquivo durante o algoritmo
        startTimer = clock();
        {
            switch(mode)
            {
                //Modo de leitura Large: verifica se as palavras sao palindromos e se representam numeros primos
                case 'L':
                    startTimer = clock();
                    //Gera a lista de numeros primos (de 2 a 20000)
                    sievePrimeNumbers(&primeList, 20000);
                    stopTimer = clock();
                    printf("Tempo Criacao Crivo: %lf\n", (double)(stopTimer-startTimer)/CLOCKS_PER_SEC);
                    
                    while(ret != 0)
                    {
                        //Faz a leitura da proxima palavra
                        //Qualquer tipo de simbolo eh descartado no processo
                        ret = readWordFromFile(inputFile, word);
                        
                        //Somente se a palava tiver 3 caracteres ou mais sera aceito como um palindromo
                        if(ret == 1)
                        {
                            //Verifica se e' palindromo
                            if(isPalindrome(word, &isPalCount))
                            {
                                //Adiciona o palindromo ao vector de palindromos
                                addPalindrome(&palindromes, word, &primeList, mode, &addPalCount, &isPrimeCount);
                                wpCount++;
                            }
                        }
                    }
                    break;
                    
                //Modo de leitura Small: verifica se as palavras e sentencas sao palindromos
                case 'S':
                    while(ret != 0)
                    {
                         //Faz a leitura da proxima frase (terminando em ".!?\n\r")
                        ret = readSentenceFromFile(inputFile, sentence, &palindromes, &primeList, &isPalCount, &addPalCount, &isPrimeCount);
                        
                        //Somente se a palava tiver 3 caracteres ou mais sera aceito como um palindromo
                        if(ret == 1)
                        {
                            //Verifica se e' palindromo
                            if(isPalindrome(sentence, &isPalCount))
                            {
                                //Adiciona o palindromo ao vector de palindromos
                                addPalindrome(&palindromes, sentence, &primeList, mode, &addPalCount, &isPrimeCount);
                                spCount++;
                            }
                        }
                    }
                    break;
                    
                default:
                    printf("Invalid reading mode. Provide two options only <S>(small) or <L>(large)\n");
                    exit(1);
                    break;
            }
        }
    }
    else
    {
        printf("Failed to execute. Supply an input file with the following format:\n%s\n",
               "$ ./palindromeCheck input.txt <S,L>");
        exit(1);
    }
    stopTimer = clock();
    printf("Tempo Processar Arquivo+Palindromos: %lf\n", (double)(stopTimer-startTimer)/CLOCKS_PER_SEC);
    
    //Imprime todos os palindromos encontrados!
    startTimer = clock();
    for(int i = 0; i < palindromes.size(); i++)
    {
        printf("%s - %d occurrences", palindromes[i].word.c_str(), palindromes[i].count);
        
        if(palindromes[i].primeNumber != 0)
            printf(" - prime number %d", palindromes[i].primeNumber);
        
        printf("\n");
    }
    printf("Word Palindromes: %d (%lu unique)\n", wpCount, palindromes.size());
    printf("Sentence Palindromes: %d\n", spCount);
    
    stopTimer = clock();
    printf("Tempo Impressao dos Resultados: %lf\n", (double)(stopTimer-startTimer)/CLOCKS_PER_SEC);
    
    printf("isPal: %d\naddPal: %d\nisPrime: %d\n", isPalCount, addPalCount, isPrimeCount);
    
    if(strlen(fileBuffer) > 0)
        free(fileBuffer);
    
    fclose(inputFile);
    
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

int readSentenceFromFile(FILE *fp, char *buffer, vector<Palindrome> *palindromes, vector<int> *primeList, int *isPalCount, int *addPalCount, int *isPrimeCount)
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
                //printf("%d - %s\n", i, word);
                
                if(isPalindrome(word, isPalCount))
                {
                    wpCount++;
                    addPalindrome(palindromes, word, primeList, 'S', addPalCount, isPrimeCount);
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

bool isPalindrome(const char *input, int *callCount)
{
    (*callCount)++;
    
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

int addPalindrome(vector<Palindrome> *palindromes, string word, vector<int> *primeList, char mode, int *callCount, int *isPrimeCount)
{
    (*callCount)++;
    
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
            
            (*isPrimeCount)++;
            
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


