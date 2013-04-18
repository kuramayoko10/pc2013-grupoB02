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
#include "primeNumber.h"

bool isPalindrome(const char *input);


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
        fscanf(fileSmall, "%s ", readBuffer);
        
        if(isPalindrome(readBuffer))
            printf("%s\n", readBuffer);
    }
    
    return 0;
}

bool isSymbol(char c)
{
    
}

bool isPalindrome(const char *input)
{
    unsigned int size = (unsigned)strlen(input) - 1;
    int i = 0;
    
    while(i <= size)
    {
        
        
        if(input[i] != input[size-i])
            return false;
    }
    
    return true;
}








