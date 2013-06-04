#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "common.h"
#include "hmap.h"
#include "qrand.h"
#include "process.h"


char **less_array, **more_array;
struct hmap *less_map;



int main(void)
{
	qrand_seed((unsigned)time(NULL));
	init();
	process_less();
	process_more();
	end();
	return SUCCESS;
}

/* 	Calculates the hash of a null terminated string by using some magic
 * numbers.
 */
unsigned long string_hash(const void *a)
{
	const unsigned char *str = a;
	unsigned long hash = 5381;
	int c;
	while ((c = *str++))
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	return (hash)%MAP_SIZE;
}

int word_compound(char **array, int threshold, char *finalWord)
{
	int i, j;
	char *concat = (char*)malloc(sizeof(char)*40);
	for(i = 0; i < threshold; i++)
	{
		if(array[i][0] == finalWord[0])
		{
			for(j = 0; j < threshold; j++)
			{
				strcpy(concat, array[i]);
				strcat(concat, array[j]);
				if(strcmp(concat, finalWord) == 0)
				{
					return 1;
				}
			}
		}
	}

	return 0;
}

void init(void)
{
	unsigned i;                                                        
	bool flag_found=FALSE;                                             
	char input_string[40];
	FILE *less_file, *more_file;
	less_map = hmap_init(MAP_SIZE, 6*sizeof(char), sizeof(bool), string_hash); 
	less_file = fopen("less.txt", "r");                                
	more_file = fopen("more.txt", "r");                                
	less_array = malloc(sizeof(char *)*N_LESS_WORD);                   
	more_array = malloc(sizeof(char *)*N_MORE_WORD);                   
	for (i=0, memset(input_string, '\0', 6);                           
			fscanf(less_file, "%s", input_string) != EOF; i++) 
	{                                                                  
		less_array[i] = (char*)malloc(sizeof(char)*6);             
		strcpy(less_array[i], input_string);                       
		hmap_insert(less_map, less_array[i], &flag_found);        
		memset(input_string, 0, 6);
	}
	for (i=0; fscanf(more_file, "%s", input_string) != EOF; i++)   
	{
		more_array[i] = (char*)malloc(sizeof(char)*41);            
		strcpy(more_array[i], input_string);                       
	}
	fclose(less_file);                                                 
	fclose(more_file);                                                 
}       

void end(void)
{
	unsigned i;
	hmap_free(less_map);                                              
	for(i = 0; i < N_LESS_WORD; i++)                                  
		free(less_array[i]);                                   
	for(i = 0; i < N_MORE_WORD; i++)                               
		free(more_array[i]);                                
	free(less_array);                                              
	free(more_array);                                           
}


void process_less(void)
{
	unsigned i;
	char rand_str[6];
	bool found_flag;
	for (i=0; i<N_LESS_WORD;)
	{
		qrand_word(rand_str);
		if (hmap_remove(less_map, rand_str, &found_flag)==SUCCESS)
		{
			printf("Found:%s\n", rand_str);
			i++;
		}
	}
}

void process_more(void)
{
	unsigned i;
	for(i = 0; i < N_MORE_WORD; i++)
	{
		if(word_compound(less_array, N_LESS_WORD, more_array[i]))
		{
			printf("Found: %s\n", more_array[i]);
		}
	}
}

