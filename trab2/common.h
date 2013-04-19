#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <string>

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

#ifndef __cplusplus
typedef unsigned char bool;
#endif

typedef struct
{
    std::string word;
    unsigned int count;
}WordCount;

void sievePrimeNumbers(std::vector<int> *primeList, unsigned int limit);
void loadList(std::vector<int> *primeList, const char* filename);
void saveList(std::vector<int> *primeList, const char* filename);
bool isPrimeNumber(std::vector<int> *primeList, unsigned int number);

#endif
