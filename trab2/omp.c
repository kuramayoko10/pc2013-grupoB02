#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define STATE_ALNUM 0
#define STATE_SPACE 1
#define STATE_PUNCT 2


int main(int argc, char **argv) {

	FILE *file;
	char *text;
	char *phrase, *word;
	unsigned word_size, phrase_size, cur_state; 
	register unsigned i, j;
	unsigned div_num, block_size;
	unsigned word_palin_count=0, phrase_palin_count=0, prime_count=0;
	clock_t begin, end;
	char mode;
	/* div_num é o numero de iteracoes do loop a ser executado*/
	div_num = omp_get_num_procs();
	if (argc != 2)
		return 0;
	/* No caso do processamento ser do arquivo grande. */
	if (strcmp(argv[1], "big_text") == 0)
	{
		mode = BIG_MODE;
		block_size = BIG_TEXT_SIZE/div_num;	
		file = fopen(argv[1], "r");
		if (file == NULL)
			return FAILURE;
		text = malloc(BIG_TEXT_SIZE*sizeof(char));
		for (i=0; i<BIG_TEXT_SIZE; i++)
			text[i] = tolower(getc(file));
	}
	/* No caso do pequeno. */
	else if (strcmp(argv[1], "small_text") == 0)
	{
		mode = SMALL_MODE;
		block_size = SMALL_TEXT_SIZE/div_num;	
		file = fopen(argv[1], "r");
		if (file == NULL)
			return FAILURE;
		text = malloc(SMALL_TEXT_SIZE*sizeof(char));
		for (i=0; i<SMALL_TEXT_SIZE; i++)
			text[i] = tolower(getc(file));
	}
	/* caso nao seja passado ennhum arquivo, sai da execução. */
	else return 0;
	fclose (file);
    
	/* For each block*/ 
	begin = clock();
#pragma omp parallel for private(i, j, word_size, phrase_size, word, phrase, cur_state)
	/* O texto inteiro é dividido em div_num pedaçoes de block_size cada.
Cada thread processara até o fim da ultima frase do bloco. O processamento se 
da por meio de uma automato finito com 3 estados, cada estado representa o 
ultimo caracter lido do bloco, dependendo do caracter que será lido ocorrerá 
uma transicao. Cada transiçao tem associada a si diversas açoes.*/
	for (i=0; i<div_num; i++)
	{
		cur_state = STATE_PUNCT;
		phrase = text+i*block_size;
		word = phrase;
		if (i != 0) /* Finds next phrase beggining */
			for (j=0; j<block_size&&(isspace(phrase[j])||isalnum(phrase[j])); j++);	
		word_size=1, phrase_size=1;
		for (j=0; j<block_size||cur_state!=STATE_PUNCT; j++)
		{
			if (isalnum(phrase[phrase_size])) /* Case char is alphanumeric */
			{
				if (cur_state == STATE_ALNUM)
				{
					++word_size;
					++phrase_size;
				}
				else if (cur_state == STATE_SPACE)
				{
					++word_size;
					++phrase_size;
				}
				else 
				{
					++word_size;
					++phrase_size;
				}
				cur_state = STATE_ALNUM;
			}
			else if (isspace(phrase[phrase_size])) /* Case char is a space */
			/* or \n */
			{
				if (cur_state == STATE_ALNUM)
				{
					if (word_is_palin(word, word_size))
					{
#pragma omp atomic
						++word_palin_count;
						if (is_prime(word_sum(word, word_size), mode))
#pragma omp atomic
							++prime_count;
					}
					word+=word_size+1;
					word_size=0;
					++phrase_size;
				}
				else if (cur_state == STATE_SPACE)
				{
					++word;
					++phrase_size;
				}
				else 
				{
					++word;
					++phrase_size;	
				}
				cur_state = STATE_SPACE;
			}
			else /* Case char is a ponctuator like: .;"',?! */
			{
				if (cur_state == STATE_ALNUM)
				{
					if (word_is_palin(word, word_size))
					{
#pragma omp atomic
						++word_palin_count;
						if (is_prime(word_sum(word, word_size), mode))
#pragma omp atomic
							++prime_count;
					}
					if (phrase_is_palin(phrase, phrase_size, mode))
#pragma omp atomic
							++phrase_palin_count;
					word+=word_size+1;
					phrase+=phrase_size+1;
					word_size=0;
					phrase_size=0;
				}
				else if (cur_state == STATE_SPACE)
				{ if (phrase_is_palin(phrase, phrase_size, mode))
						++phrase_palin_count;
					phrase+=phrase_size+1;
					phrase_size=0;
					word = phrase;
				}
				else 
				{
					++word;
					++phrase;
				}
				cur_state = STATE_PUNCT;
			}
		}
	}
	end = clock();
	printf("Took %f seconds.\n ", (float) (end-begin)/CLOCKS_PER_SEC);
	printf("wp: %u pp: %u p: %u\n", word_palin_count, phrase_palin_count, 
			prime_count);
	free(text);
	return SUCCESS;
}
