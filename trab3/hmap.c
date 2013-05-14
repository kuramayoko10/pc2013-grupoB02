#include "hmap.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool is_empty(void *data, unsigned long size)
{
	register unsigned long i;
	for (i=0; i<size; i++)
	{
		if (((unsigned char *)data)[i] != HMAP_EMPTY_BLOCK)
            return FALSE;
	}
	return TRUE;
}

bool is_deleted(void *data, unsigned long size)
{
	register unsigned long i;
	for (i=0; i<size; i++)
	{
		if (((unsigned char *) data)[i] != HMAP_DELETED_BLOCK)
			return FALSE;
	}
	return TRUE;
}

bool is_reserved(void *data, unsigned long size)
{
	register unsigned long i;
	if (((unsigned char *) data)[0] != HMAP_EMPTY_BLOCK && 
			((unsigned char *) data)[0] != HMAP_DELETED_BLOCK)
		return FALSE;
	for (i=0; i<size; i++)
	{
		if (((unsigned char *) data)[i] != ((unsigned char *) data)[0])
			return FALSE;
	}
	return TRUE;
}

unsigned long std_hash(const void *a)
{
	unsigned long hash = 5831;
	int c, *n;
	n = (int *) a;
	while ((c = *n++))
		hash = ((hash<<5) + hash) + c;
	return hash;
}

struct hmap *hmap_init(unsigned long map_size, unsigned long data_size,
		unsigned long key_size, unsigned long (*hash)(const void *))
{
	struct hmap *map;
	map = malloc(sizeof(struct hmap));
	if (map == NULL)
		return NULL;
	map->keys = malloc(map_size*key_size);
	if (map->keys == NULL)
	{
		free(map);
		return NULL;
	}
	map->data = malloc(map_size*data_size);
	if (map->data == NULL)
	{
		free(map->keys);
		free(map);
		return NULL;
	}
  memset(map->keys, HMAP_EMPTY_BLOCK, map_size*key_size);
	map->map_size = map_size;
	map->key_size = key_size;
	map->data_size = data_size;
	if (hash == NULL)
		map->hash = std_hash;
	else
		map->hash = hash;
	return map;
}

void hmap_free(struct hmap *map)
{
	free(map->keys);
	free(map->data);
	free(map);
}

int hmap_insert(struct hmap *map, void *key, void *data)
{
	unsigned long i=0, index=map->hash(key)%map->map_size;
	unsigned char *aux_key=map->keys, *aux_data=map->data;
	if (is_reserved(key, map->key_size))/*Makes sure to dont add reserved keys*/
        return FAILURE;
	while (!is_reserved(aux_key+((index+i)%map->map_size)*(map->key_size),
			map->key_size))
	/*While haven't found a empty or deleted key*/
	{
		if (i==map->map_size) /*If already passed through the whole map*/
			return FAILURE;
		i++;
	}
	memcpy(aux_key+((index+i)%map->map_size)*(map->key_size), key,
			map->key_size); /*Write key and data*/
	memcpy(aux_data+((index+i)%map->map_size)*(map->data_size), data,
			 map->data_size);
	return SUCCESS;
}

int hmap_remove(struct hmap *map, void *key, void *data)
{
	unsigned long i=0, index=map->hash(key)%map->map_size;
	unsigned char *aux_key=map->keys, *aux_data=map->data;
	while (!is_empty(aux_key+((index+i)%map->map_size)*(map->key_size),
				map->key_size)&&i!=map->map_size)
	{
		if (memcmp(aux_key+((index+i)%map->map_size)*(map->key_size), 
				key, map->key_size) == 0)
		{
			memcpy(data, aux_data+((index+i)%map->map_size)*
					(map->data_size), map->data_size);
			memset(aux_key+((index+i)%map->map_size)*(map->key_size)
					, 0xff,	map->key_size);
			return SUCCESS;
		}
		i++;
	}
	return FAILURE;
}

int hmap_search(struct hmap *map, void *key, void *data)
{
	unsigned long i=0, index=map->hash(key)%map->map_size;
	unsigned char *aux_key=map->keys, *aux_data=map->data;
	while (!is_empty(aux_key+((index+i)%map->map_size)*(map->key_size),
				map->key_size)&&i!=map->map_size)
	{
		if (memcmp(aux_key+((index+i)%map->map_size)*(map->key_size), 
				key, map->key_size) == 0)
		{
			memcpy(data, aux_data+((index+i)%map->map_size)*
					(map->data_size), map->data_size);
			return SUCCESS;
		}
		i++;
	}
	return FAILURE;
}

void hmap_print(struct hmap *map)
{
  unsigned long i, j;
	unsigned char *aux_key=map->keys, *aux_data=map->data;
  printf("Struct hmap at %p\n", (void *) map);
  printf("map_size=%lu    key_size=%lu    data_size=%lu\n", map->map_size,
        map->key_size, map->data_size);
	for (i=0; i<map->map_size; i++)
	{
  	if (is_empty(aux_key+i*map->key_size,
				map->key_size))
			printf("Block %lu: empty\n", i);
		else if (is_deleted(aux_key+i*map->key_size,
				map->key_size))
			printf("Block %lu: deleted\n", i);
		else
		{
			printf("Block %lu: key = ", i);
			for (j=0; j<map->key_size; j++)
				printf("%u ", aux_key[i*map->key_size+j]);
			printf("data = ");
			for (j=0; j<map->key_size; j++)
				printf("%u ", aux_data[i*map->data_size+j]);
			printf("\n");
		}
	}
}





