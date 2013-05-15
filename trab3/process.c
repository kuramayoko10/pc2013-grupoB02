#include "common.h"
#include "hmap.h"
#include "qrand.h"
#include "process.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>


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

int read_until_comma(FILE *file, char *string, unsigned string_size)
{
	unsigned i;
	char c;
	for (i=0; i<string_size-1; ++i)
	{
		c = tolower(getc(file));
		if (c == ' ')
		{
			--i;
			continue;
		}
		if (c == ',')
		{ string[i] = '\0';
			return SUCCESS;		
		}
		if (c == EOF)
		{
			string[i] = '\0';
			return FAILURE;
		}
		string[i] = c;
	}
	string[i] = '\0';
	return SUCCESS;
}

int main(void)
{
	char input_string[30];
	bool flag_found;
	FILE *input_file;
	struct hmap *word_map;
	flag_found = FALSE;
	input_file = fopen("palavras.txt", "r");
	word_map = hmap_init(MAP_SIZE, 30*sizeof(char), sizeof(bool),
			 string_hash);
	if (word_map == NULL)
		return FAILURE;
	while (read_until_comma(input_file, input_string, 30) == SUCCESS)
	{
		if (hmap_search(word_map, input_string, &flag_found) == 
				FAILURE)
			hmap_insert(word_map, input_string, &flag_found);
	}
	hmap_print(word_map);
	hmap_free(word_map);
	fclose(input_file);
	return SUCCESS;
}
