// primeNumber - Trabalho 2
// Deteccao de numeros primos
// Geracao de lista de numeros primos atraves do algoritmos
// do Crivo de Eratostenes
//
// Grupo: 02
// Cassiano K. Casagrande, Guilherme S. Gibertoni, Rodrigo V. C. Beber

#include "primeNumber.h"



void sievePrimeNumbers(std::vector<int> *primeList, unsigned int limit)
{
    unsigned int lastCheck = sqrt(limit);
    unsigned int check = 2;
    unsigned int curPos = 0;
    
    //Preenche o vetor inicial
    for(int i = 0; i < limit-1; i++)
    {
        primeList->push_back(i+2);
    }
    
    //Itera eliminando os multiplos de numeros primos conhecidos ate encontrar lastCheck
    while(check <= lastCheck)
    {
        for(int i = curPos+1; i < primeList->size(); i++)
        {
            if(primeList->at(i) % check == 0)
                primeList->erase(primeList->begin()+i);
        }
        
        check = primeList->at(++curPos);
    }
}

bool isPrimeNumber(std::vector<int> *primeList, unsigned int number)
{
    unsigned int curPos, left, right, curValue;
    
    left = 0;
    right = (unsigned)primeList->size()-1;
    
    while(left <= right)
    {
        curPos = (left+right)/2;
        curValue = primeList->at(curPos);
        
        if(number == curValue)
            return true;
        else if(number < curValue)
        {
            right = curPos-1;
        }
        else //number > primeList[curPos]
        {
            left = curPos+1;
        }
    }
    
    
    return false;
}

void saveList(std::vector<int> *primeList, const char* filename)
{
    FILE *file = fopen(filename, "w+");
    int i = 0;
    
    while(i < primeList->size())
    {
        fprintf(file, "%d\n", primeList->at(i++));
    }
    
    fclose(file);
}

void loadList(std::vector<int> *primeList, const char* filename)
{
    FILE *file = fopen(filename, "r");
    
    //char read[16];
    int value = 0;
    
    while(!feof(file))
    {
        fscanf(file, "%d ", &value);
        primeList->push_back(value);
    }
    
    fclose(file);
}

//Testes
/*int main(int argc, char *argv[])
 {
 unsigned int test = 113;
 
 //Gera uma nova lista de numeros primos ate o limite abaixo
 sievePrimeNumbers(5000);
 //Salva a lista em arquivo
 //saveList("prime-numbers.txt");
 
 //Carrega a lista salva
 //loadList("prime-numbers.txt");
 //Testa se um dado numero e' primo
 if(isPrimeNumber(test))
 printf("O numero %d e' primo\n", test);
 else
 printf("O numero %d NAO e' primo\n", test);
 
 return 0;
 }*/






