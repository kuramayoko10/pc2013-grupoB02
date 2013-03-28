#include "common.h"
#include <stdio.h>

int main(int argc, char **argv)
{
	register unsigned i=0;
	char c1, c2;
	FILE *f1, *f2;
	if (argc < 3)
		return FAILURE;
	f1 = fopen(argv[1], "r");
	if (f1 == NULL)
		return FAILURE;
	f2 = fopen(argv[2], "r");
	if (f2 == NULL)
	{
		fclose(f1);
		return FAILURE;
	}
	while ((c1 = getc(f1)) != EOF && (c2 = getc(f2)) != EOF)
	{
		if (c1 != c2)
			break;
		++i;
	}
	printf("First non equal character is at %u.\n", i);
	fclose(f1);
	fclose(f2);
	return SUCCESS;
}
