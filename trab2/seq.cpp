// sequencial.cpp - Trabalho 2
// Deteccao de palindromos em textos
// Algoritmo sequencial e sem paralelizacao
//
// Grupo: 02
// Cassiano K. Casagrande, Guilherme S. Gibertoni, Rodrigo V. C. Beber

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "common.h"

bool isPalindrome(const char *input);
bool isSymbol(char input);
int readWordFromFile(FILE *fp, char *buffer);


int main(int argc, const char * argv[])
{
    std::vector<int> primeList;
    FILE *fileSmall, fileLarge;
    char readBuffer[32];
    
    //Gera a lista de numeros primos
    //sievePrimeNumbers(&primeList, 5000);
    
    //Carrega a lista de numeros primos
    loadList(&primeList, "prime-numbers.txt");
    
    fileSmall = fopen("shakespe.txt", "r");
    
    //Leitura por palavra
    printf("Palindrome - Word Check\n");
    while(!feof(fileSmall))
    {
        //Faz a leitura da proxima palavra, descartando pontuacoes
        readWordFromFile(fileSmall, readBuffer);
        
        if(isPalindrome(readBuffer))
            printf("%s\n", readBuffer);
    }
    
    return 0;
}


int readWordFromFile(FILE *fp, char *buffer)
{
    char read;
    int i = 0;
    
    buffer[0] = '\0';
    read = fgetc(fp);
    
    while(!isSymbol(read) || !isspace(read))
    {
        buffer[i++] = read;
        read = fgetc(fp);
    }
    buffer[i] = '\0';
    
    if(strlen(buffer) > 0)
        return 1;
    
    return 0;
}

bool isSymbol(char input)
{
    //if( (input >= 33 && input <= 47) || (input >= 58 && input <= 64) || (input >= 91 && input <= 96) || (input >= 123 && input <= 126))
    if((input >= 48 && input <= 57) || (input >= 65 && input <= 90) || (input >= 397 && input <= 122))
    {
        return 0;
    }
    
    return 1;
}

bool isPalindrome(const char *input)
{
    int size = (unsigned)strlen(input) - 1;
    int i = 0;
    
    while(i <= size)
    {
        if(size < 0)
            return false;
            
        if(input[i] != input[size-i])
            return false;
    }
    
    return true;
}








