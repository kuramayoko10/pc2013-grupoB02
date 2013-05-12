// primeNumber - Trabalho 2
// Deteccao de numeros primos
// Geracao de lista de numeros primos atraves do algoritmos
// do Crivo de Eratostenes
//
// Grupo: 02
// Cassiano K. Casagrande, Guilherme S. Gibertoni, Rodrigo V. C. Beber

#include "common.h"


/* Gera a lista de numeros primos limitada por limits e armazena no vector primeList. A geracao eh baseada no Crivo de Eratostenes
 * primeList: vector para armazenar os numeros primos
 * limit: inteiro para limitar o maior numero primo da lista
 *
 * return: void
 */
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

/* Verifica se um dado numero e' primo
 * primeList: vector que tem armazenado os numeros primos
 * number: inteiro positivo que deseja-se verificar
 *
 * return: true se for primo. false caso contrario
 */
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

/* Salva a lista de numeros primos em um arquivo
 * primeList: vector que armazena os numeros primos
 * filename: nome do arquivo a ser gerado
 *
 * return: void
 */
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

/* Carrega a lista de numeros primos de um arquivo para um vector
 * primeList: vector para armazenar os numeros primos
 * filename: nome do arquivo que contem os numeros primos. Os numero deverao constar sozinhos em cada linha.
 *
 * return: void
 */
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






