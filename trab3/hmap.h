#ifndef HMAP_H
#define HMAP_H

#include "common.h"

#define HMAP_EMPTY_BLOCK 0xff
#define HMAP_DELETED_BLOCK 0xfe


/*
*	@section Description A generic hash map data structure implementation,
* using a probing collision solving. The hash map stores data with data_size 
* bytes each identified by a key with key_size bytes and can 
* store up to map_size blocks.
* 	There are reserved keys that are used to mark empty and deleted map 
* positions. The reserved keys contain only 0xff and 0xfe bytes inside its key.
* 	The hash map keeps the keys and respective data on separated arrays.
*/
struct hmap
{
	unsigned long map_size;
	unsigned long data_size;
	unsigned long key_size;	
	void *keys;
	void *data;
	unsigned long (*hash)(const void *);
};



/*
* 	Inits the a hash map structure and returns a pointer to it.
* 	@param map_size Size of the hash map.
* 	@param data_size Size of the data that will be stored in the map.
* 	@param key_size Size of the key that will be used to identify each data.
* 	@param hash Pointer to a function that calculates a key hash, if a NULL
* pointer is passed, the hash function will be a standart defined hash.
* 	@return Returns a pointer to the created hmap structure. If there isn't
* enough memory, returns a NULL pointer. 
*/
struct hmap *hmap_init(unsigned long map_size, unsigned long key_size, 
	unsigned long data_size, unsigned long (*hash)(const void *)); 

/*
*		Frees all memory used by the hmap structure.
* 	@param map hmap structure that will be freed.
*/
void hmap_free(struct hmap *map);

/*
* 	Inserts a new data identified by a key on a given hmap structure.
*	@param map Pointer to a hmap structure where data is going to be added.
* 	@param key Pointer to the key which identifies the data.
* 	@param data Pointer to the data that will be added on hmap.
* 	@return Return SUCCESS if the insertion was successfull, if there 
* wasn't space on the map returns FAILURE.
*/
int hmap_insert(struct hmap *map, void *key, void *data);

/*
* 	Removes a data identified by key on a given hmap strucuture.
* 	@param map Pointer to a hmap structure from where the data is going to
* be removed.
* 	@param key Pointer to a key that identifies the data to be removed 
* 	@param data Pointer to which the removed data will be writen before
* removal.
* 	@return Returns SUCCESS if the data was found and the removal was 
* successfull and FAILURE otherwise.
*/
int hmap_remove(struct hmap *map, void *key, void *data);

/*
* 	Searches a data identified by key on a given hmap strucuture.
* 	@param map Pointer to a hmap structure from where the data is going to
* be searched.
* 	@param data Pointer to which the data data will be writen.
* 	@return Returns SUCCESS if the data was found  and FAILURE otherwise.
*/
int hmap_search(struct hmap *map, void *key, void *data);

/*
*	Prints all data in the hmap structure for debugging.
* 	@param map Pointer to the hmap structure that will be printed.
*/

int hmap_update(struct hmap *map, void *key, void *data);

void hmap_print(struct hmap *map);


#endif
