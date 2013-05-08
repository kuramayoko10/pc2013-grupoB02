// sequencial.cpp - Trabalho 2
// Deteccao de palindromos em textos
// Algoritmo sequencial e sem paralelizacao
//
// Grupo: 02
// Cassiano K. Casagrande, Guilherme S. Gibertoni, Rodrigo V. C. Beber

#include "common.h"

using namespace std;

bool isPalindrome(const char *input);
bool isSymbol(char input);
bool endOfSentence(char input);
int readWordFromFile(FILE *fp, char *buffer);
int readWordFromBuffer(char *buffer, char *word);
int readSentenceFromFile(FILE *fp, char *buffer, vector<Palindrome> *palindromes, vector<int> *primeList);
int readSentenceFromBuffer(char *buffer, char *sentence, vector<Palindrome> *palindromes, vector<int> *primeList);
int addPalindrome(vector<Palindrome> *palindromes, string word, vector<int> *primeList, char mode);
int sumASCII(const char *str);
void readFileToBuffer(FILE *fp, char *buffer, int size);


int main(int argc, const char * argv[])
{
    vector<int> primeList;
    vector<Palindrome> palindromes;
    FILE *inputFile;
    char smallBuffer[64];
    char bigBuffer[2048];
    char mode = 'a';
    char *fileBuffer;
    //char fileBuffer[6000000];
    clock_t startTimer, stopTimer;
    int ret = 0;
    
    //Gera a lista de numeros primos
    startTimer = clock();
    sievePrimeNumbers(&primeList, 20000);
    stopTimer = clock();
    //saveList(&primeList, "prime-numbers.txt");
    
    printf("Tempo Criacao Crivo: %lf\n", (double)(stopTimer-startTimer)/CLOCKS_PER_SEC);
    
    if(argc >= 2)
    {
        inputFile = fopen(argv[1], "r");
        
        if(!inputFile)
        {
            printf("File not found!\n");
            exit(1);
        }
        
        if(strlen(argv[2]) == 1)
            mode = argv[2][0];
        
        //Buffer de 64MB
        startTimer = clock();
        fileBuffer = (char*)malloc(sizeof(char)*6000000);
        readFileToBuffer(inputFile, fileBuffer, 5400000);
        stopTimer = clock();
        printf("Tempo Processar Arquivo: %lf\n", (double)(stopTimer-startTimer)/CLOCKS_PER_SEC);
        
        //Leitura por palavra
        startTimer = clock();
        //while(!feof(inputFile))
        {
            switch(mode)
            {
                case 'L':
                    //Faz a leitura da proxima palavra, descartando pontuacoes
                    /*if(readWordFromFile(inputFile, smallBuffer))
                    {
                        if(isPalindrome(smallBuffer))
                        {
                            //printf("%s\n", smallBuffer);
                            
                            //Adiciona o palindromo ao vector de palindromos
                            int pos = addPalindrome(&palindromes, smallBuffer, NULL, 'S');
                        }
                    }*/
                    while((ret = readWordFromBuffer(fileBuffer, smallBuffer)) != 0)
                    {
                        if(ret == 1)
                            if(isPalindrome(smallBuffer))
                            {
                                //Adiciona o palindromo ao vector de palindromos
                                int pos = addPalindrome(&palindromes, smallBuffer, NULL, 'S');
                            }
                    }
                    break;
                    
                case 'S':
                    //Faz a leitura da proxima frase (terminando em ".!?\n"
                    //while(readSentenceFromFile(inputFile, bigBuffer, &palindromes, &primeList))
                    while((ret = readSentenceFromBuffer(fileBuffer, bigBuffer, &palindromes, &primeList)) != 0)
                    {
                        //printf("%s\n", bigBuffer);
                        
                        if(ret == 1)
                            if(isPalindrome(bigBuffer))
                            {
                                int pos = addPalindrome(&palindromes, bigBuffer, &primeList, 'L');
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
    printf("Tempo Processar Palindromos: %lf\n", (double)(stopTimer-startTimer)/CLOCKS_PER_SEC);
    
    startTimer = clock();
    printf("Palindrome - %s\n", argv[1]);
    for(int i = 0; i < palindromes.size(); i++)
    {
        printf("%s - %d occurrences", palindromes[i].word.c_str(), palindromes[i].count);
        
        if(palindromes[i].primeNumber != 0)
            printf(" - prime number %d", palindromes[i].primeNumber);
        
        printf("\n");
    }
    stopTimer = clock();
    
    printf("Tempo Impressao dos Resultados: %lf\n", (double)(stopTimer-startTimer)/CLOCKS_PER_SEC);
    
    fclose(inputFile);
    
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
        return 1;
    
    return 0;
}

int readWordFromBuffer(char *buffer, char *word)
{
    static int i = 0;
    int w = 0;
    char read;
    
    read = buffer[i];
    while(!isSymbol(read) && !isspace(read))
    {
        word[w++] = read;
        read = buffer[++i];
    }
    word[w] = '\0';
    i++;
    
    //printf("i: %d\n", i);
    
    if(i >= strlen(buffer))
        return 0;
    
    //Consideramos palindromos apenas as palavras de mais de 3 caracteres.
    //Pois estas tem um significado claro na lingua
    //Artigos 'a' ou palavras comprimidas "we'll" nao nos interessa
    if(strlen(word) > 2)
        return 1;
    
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
                if(isPalindrome(word))
                    addPalindrome(palindromes, word, primeList, 'L');
                
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
    
    if(feof(fp))
        return 0;
    
    //Consideramos palindromos apenas as palavras de mais de 3 caracteres.
    //Pois estas tem um significado claro na lingua
    //Artigos 'a' ou palavras comprimidas "we'll" nao nos interessa
    if(strlen(buffer) > 2)
        return 1;
    
    return 2;
}

int readSentenceFromBuffer(char *buffer, char *sentence, vector<Palindrome> *palindromes, vector<int> *primeList)
{
    char read;
    char word[128] = "\0";
    static int i = 0;
    int s = 0;
    int w = 0;
    
    sentence[0] = '\0';
    read = buffer[i];
    
    //printf("size: %lu\n", strlen(buffer));
    
    while(!endOfSentence(read) && i < strlen(buffer))
    {
        if(isSymbol(read) || isspace(read))
        {
            //Processa a palavra anterior e descarta a pontuacao/espaco_branco
            word[w] = '\0';
            
            if(w >= 3)
            {
                if(isPalindrome(word))
                    addPalindrome(palindromes, word, primeList, 'L');
                
                //printf("%s\n", word);
            }
            
            w = 0;
        }
        else
        {
            sentence[s++] = read;
            word[w++] = read;
        }
        
        read = buffer[++i];
    }
    sentence[s] = '\0';
    i++;
    
    printf("I: %d\n", i);
    
    if(i >= strlen(buffer))
        return 0;
    
    //Consideramos palindromos apenas as palavras de mais de 3 caracteres.
    //Pois estas tem um significado claro na lingua
    //Artigos 'a' ou palavras comprimidas "we'll" nao nos interessa
    if(strlen(sentence) > 2)
        return 1;
    
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
            
            clock_t startTimer = clock();
            if(isPrimeNumber(primeList, num))
            {
                clock_t stopTimer = clock();
                printf("Tempo Encontrar Primo: %lf\n", (double)(stopTimer-startTimer)/CLOCKS_PER_SEC);
                
                palindromes->at(i-1).primeNumber = num;
                
                printf("%s\n", word.c_str());
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

void readFileToBuffer(FILE *fp, char *buffer, int size)
{
    //char *read = (char*)malloc(sizeof(char)*1048576);
    int i = 0;
    
    //while(!feof(fp))
    while(i < size)
    {
        buffer[i++] = fgetc(fp);
    }
    buffer[i] = '\0';
}

