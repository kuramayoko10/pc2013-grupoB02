#include "common.h"
#include "hmap.h"
#include "qrand.h"
#include "process.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

unsigned long string_hash(const void *a)
/* 	Calculates the hash of a null terminated string by using some magic
* numbers.
*/
{
	const unsigned char *str = a;
	unsigned long hash = 5381;
	int c;
	while ((c = *str++))
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	return (hash)%MAP_SIZE;
}

int main(void)
{
	char input_string[40];
	bool flag_found;
	FILE *input_file;
	struct hmap *word_map;
	unsigned long i=0, sum=0, rep=0;
	qrand_seed(time(NULL));
	flag_found = FALSE;
	input_file = fopen("palavras.txt", "r");
	word_map = hmap_init(MAP_SIZE, 40*sizeof(char), sizeof(bool),
			string_hash);
	if (word_map == NULL)
		return FAILURE;
	memset(input_string, '\0', 40);
	while (fscanf(input_file, "%40s", input_string) != EOF)
	{
		if (hmap_search(word_map, input_string, &flag_found) == 
				FAILURE)
		{
			hmap_insert(word_map, input_string, &flag_found);
			if (strlen(input_string)<=5)
			rep++;
		}
		memset(input_string, '\0', 40);
	}
	fclose(input_file);
	for (i=0;i<rep;)
	{
		qrand_word(input_string);
		flag_found=TRUE;
		if (hmap_remove(word_map, input_string, &flag_found) ==
				SUCCESS)
		{
			printf("Found %lu of %lu: %s\n", i, rep, input_string);
			i++;	
		}
	}
	hmap_free(word_map);
	return SUCCESS;
}
