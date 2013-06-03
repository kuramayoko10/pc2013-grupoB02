
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "common.h"
#include "hmap.h"
#include "qrand.h"
#include "process.h"

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

int wordCompound(char **array, int threshold, char *finalWord)
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
                    printf("Found: %s\n", finalWord);
                    return 1;
                }
            }
        }
    }
    
    return 0;
}

int main(void)
{
	char input_string[40];
    char *smallWordArray[N_SMALL_WORD], *compoundWordArray[N_COMPOUND_WORD];
	bool flag_found;
	FILE *input_file, *smallWordFile, *compoundWordFile;
	struct hmap *word_map;
	unsigned long i = 0, sum = 0, rep = 0;
    
	qrand_seed((unsigned)time(NULL));
    
	flag_found = FALSE;
	input_file = fopen("palavras.txt", "r");
	word_map = hmap_init(MAP_SIZE, 40*sizeof(char), sizeof(bool), string_hash);
    
	if (word_map == NULL)
		return FAILURE;
    
    /* Armazena palavras ordenadas em 2 listas
     * A primeira lista contem as palavras de 5 letras ou menos ordenadas
     * A segunda lista contem as palavras de mais de 5 letras, tambem ordenadas
     */
    smallWordFile = fopen("less.txt", "r");
    compoundWordFile = fopen("more.txt", "r");
    
    i = 0;
    while(fscanf(smallWordFile, "%s", input_string) != EOF)
    {
        smallWordArray[i] = (char*)malloc(sizeof(char)*6);
        strcpy(smallWordArray[i], input_string);
        i++;
    }
    
    i = 0;
    while(fscanf(compoundWordFile, "%s", input_string) != EOF)
    {
        compoundWordArray[i] = (char*)malloc(sizeof(char)*41);
        strcpy(compoundWordArray[i], input_string);
        i++;
    }
    
    fclose(smallWordFile);
    fclose(compoundWordFile);
    
    /*Teste de Composicao da primeira palavra de compoundWordArray*/
    if(wordCompound(smallWordArray, N_SMALL_WORD, compoundWordArray[0]))
        printf("Encontrei!!\n");
    else
        printf("Nao encontrei!!\n");
    
    /*Teste de Composicao da palavra brutelike*/
    if(wordCompound(smallWordArray, N_SMALL_WORD, compoundWordArray[6003]))
        printf("Encontrei!!\n");
    else
        printf("Nao encontrei!!\n");
    
    /*Armazena palavras no Hash*/
	memset(input_string, '\0', 40);
	while (fscanf(input_file, "%40s", input_string) != EOF)
	{
		if (hmap_search(word_map, input_string, &flag_found) == FAILURE)
		{
			hmap_insert(word_map, input_string, &flag_found);
            
            //Armazena o total de palavras lidas
            sum++;
			
            //Armazena o total de palavras de 5 caracteres ou menos
            if (strlen(input_string) <= 5)
                rep++;
		}
		memset(input_string, '\0', 40);
	}
	fclose(input_file);
    
	for(i = 0; i < rep; )
	{
		qrand_word(input_string);
		flag_found=TRUE;
        
		if (hmap_remove(word_map, input_string, &flag_found) == SUCCESS)
		{
			printf("Found %lu of %lu: %s\n", i, rep, input_string);
			i++;	
		}
	}
	hmap_free(word_map);
    
    for(i = 0; i < N_SMALL_WORD; i++)
        free(smallWordArray[i]);
    for(i = 0; i < N_COMPOUND_WORD; i++)
        free(compoundWordArray[i]);
    
    free(smallWordArray);
    free(compoundWordArray);
    
	return SUCCESS;
}
