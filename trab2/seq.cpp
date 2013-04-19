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
int readWordFromFile(FILE *fp, char *buffer);
void addPalindrome(vector<WordCount> *palindromes, string word);

int main(int argc, const char * argv[])
{
    vector<int> primeList;
    vector<WordCount> palindromes;
    FILE *inputFile;
    char readBuffer[32];
    char mode = 'a';
    
    //Gera a lista de numeros primos
    sievePrimeNumbers(&primeList, 5000);
    
    if(argc >= 2)
    {
        inputFile = fopen(argv[1], "r");
        
        if(!inputFile)
        {
            printf("Arquivo de entrada nao encontrado!\n");
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
                    if(readWordFromFile(inputFile, readBuffer))
                    {
                        if(isPalindrome(readBuffer))
                        {
                            //printf("%s\n", readBuffer);
                            
                            //Adiciona o palindromo ao vector de palindromos
                            addPalindrome(&palindromes, readBuffer);
                        }
                    }
                    break;
                case 'L':
                    
                    break;
                    
                default:
                    printf("Modo de leitura invalido. Aceito apenas <S>(small) ou <L>(large)\n");
                    exit(1);
                    break;
            }
        }
    }
    else
    {
        printf("Execucao invalida. Forneca o arquivo de entrada e modo de leitura:\n%s\n",
               "$ ./palindromeCheck input.txt <S,L>");
        exit(1);
    }
    
    printf("Palindrome - Word Check - %s\n", argv[1]);
    for(int i = 0; i < palindromes.size(); i++)
    {
        printf("%s - %d\n", palindromes[i].word.c_str(), palindromes[i].count);
    }
    
    //printf("Palindrome - Phrase Check - %s\n", filename);
    
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

bool isSymbol(char input)
{
    //Caracteres aceitos sao: digitos de 0-9; letras minisculas/maisculas
    if((input >= 48 && input <= 57) || (input >= 65 && input <= 90) || (input >= 97 && input <= 122))
    {
        return false;
    }
    
    return true;
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
        if(input[i] != input[size-i])
            return false;
        
        i++;
    }
    
    return true;
}

void addPalindrome(vector<WordCount> *palindromes, string word)
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
        WordCount wc;
        
        wc.word = word;
        wc.count = 0;
        
        if(i < 1)
            i = 1;
        
        palindromes->insert(palindromes->begin()+i-1, wc);
    }
}






