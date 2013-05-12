// common.h - Trabalho 2
// Deteccao de palindromos em textos
// Metodos e estruturas comuns a todas implementacoes
//
// Grupo: 02
// Cassiano K. Casagrande, Guilherme S. Gibertoni, Rodrigo V. C. Beber

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#define SUCCESS 0
#define FAILURE -1
#define TRUE 1
#define FALSE 0
#define ARRAY_SIZE(a) (sizeof(a) / sizeof(*a))

#ifdef __GNUC__
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
#define prefetch(x)			__builtin_prefetch(x)
#else
#define likely(x) 			x
#define unlikely(x)			x
#define prefetch(x)			do {}while(0);
#endif


#define BIG_TEXT_SIZE 54261766
#define SMALL_TEXT_SIZE 5344213
#define SMALL_MODE 1
#define BIG_MODE 2

#ifndef __cplusplus
typedef unsigned char bool;
#else
#include <vector>
#include <string>


typedef struct
{
    std::string word;
    unsigned int count;
    unsigned int primeNumber;
}Palindrome;

void sievePrimeNumbers(std::vector<int> *primeList, unsigned int limit);
void loadList(std::vector<int> *primeList, const char* filename);
void saveList(std::vector<int> *primeList, const char* filename);
bool isPrimeNumber(std::vector<int> *primeList, unsigned int number);
#endif

#ifdef _OPENMP

#include <omp.h>
#else
#define omp_get_num_procs(X) 1
#endif

/* Retorna TRUE se a string composta por alfa numericos apontada por s de 
tamanho n é palindroma. */
inline static bool word_is_palin(char *s, unsigned n)
{
	register int i, j;
	if (n<=3)
		return FALSE;
	for (i=0, j=n-1; i<=j; ++i, --j)
	{
		if (s[i] != s[j])
			return FALSE;
	}
	return TRUE;

}

/* Retorna um inteiro igual a soma dos valores ASCII de uma string de tamanho n
*/
inline static unsigned word_sum(char *s, unsigned n)
{
	register unsigned i, sum;
	for (i=0, sum=0; i<n; ++i, sum+=s[i]);
	return sum;
}

/* Retorna TRUE se a string composta por alfa numericos e espacos apontada por 
s de tamanho n é palindromo. */
inline static bool phrase_is_palin(char *s, unsigned n, char mode)
{
	register int i, j;
	if (n<=3||mode==BIG_MODE)
		return FALSE;
	for (i=0, j=n-1; i<=j; ++i, --j)
	{
		while (isspace(s[i])&&i<=j)
			++i;
		while (isspace(s[j])&&i<=j)
			--j;
		if (s[i] != s[j])
			return FALSE;
	}
	return TRUE;
}

/* Retorna TRUE se um numero n é primo, caso contrario retorna FALSE*/
inline static bool is_prime(unsigned n, char mode)
{ 
	register unsigned i, sqrtn = (unsigned) sqrt(n);
	if (n%2 == 0||mode==SMALL_MODE)
		return FALSE;
	for (i=3; i<sqrtn; i+=2)
		if (n%i == 0)
			return FALSE;
	return TRUE; 
}

#endif
