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
int readSentenceFromFile(FILE *fp, char *buffer, vector<Palindrome> *palindromes, vector<int> *primeList, char mode);
int addPalindrome(vector<Palindrome> *palindromes, string word, vector<int> *primeList, char mode);
int sumASCII(const char *str);

int main(int argc, const char * argv[])
{
    vector<int> primeList;
    vector<Palindrome> palindromes;
    FILE *inputFile;
    char smallBuffer[32];
    char bigBuffer[2048];
    char mode = 'a';
    
    //Gera a lista de numeros primos
    sievePrimeNumbers(&primeList, 5000);
    
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
        
        //Leitura por palavra
        while(!feof(inputFile))
        {
            switch(mode)
            {
                case 'S':
                    //Faz a leitura da proxima palavra, descartando pontuacoes
                    if(readWordFromFile(inputFile, smallBuffer))
                    {
                        if(isPalindrome(smallBuffer))
                        {
                            //printf("%s\n", smallBuffer);
                            
                            //Adiciona o palindromo ao vector de palindromos
                            int pos = addPalindrome(&palindromes, smallBuffer, NULL, 'S');
                        }
                    }
                    break;
                    
                case 'L':
                    //Faz a leitura da proxima frase (terminando em ".!?\n"
                    if(readSentenceFromFile(inputFile, bigBuffer, &palindromes, &primeList, mode))
                    {
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
    
    printf("Palindrome - %s\n", argv[1]);
    for(int i = 0; i < palindromes.size(); i++)
    {
        printf("%s - %d occurrences", palindromes[i].word.c_str(), palindromes[i].count);
        
        if(palindromes[i].primeNumber != 0)
            printf(" - prime number %d", palindromes[i].primeNumber);
        
        printf("\n");
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
        return 1;
    
    return 0;
}

int readSentenceFromFile(FILE *fp, char *buffer, vector<Palindrome> *palindromes, vector<int> *primeList, char mode)
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
                    addPalindrome(palindromes, word, primeList, mode);
                
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
    if(strlen(buffer) > 2)
        return 1;
    
    return 0;
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
        ret = word.compare(palindromes->at(i++).word);
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
                palindromes->at(i-1).primeNumber = num;
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




