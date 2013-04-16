#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

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

#define DIGITS 10000000
#define BITS_PER_DIGIT 3.32192809488736234787f

#endif


