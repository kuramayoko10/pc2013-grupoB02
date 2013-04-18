//
//  primeNumber.h
//  Trabalho2-XCode
//
//  Created by Guilherme SG on 4/17/13.
//  Copyright (c) 2013 ProgConcorrente. All rights reserved.
//

#ifndef _PRIME_NUMBER_H_
#define _PRIME_NUMBER_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>


void sievePrimeNumbers(std::vector<int> *primeList, unsigned int limit);
void loadList(std::vector<int> *primeList, const char* filename);
void saveList(std::vector<int> *primeList, const char* filename);
bool isPrimeNumber(std::vector<int> *primeList, unsigned int number);


#endif
