#ifndef QRAND_H
#define QRAND_H

#define ONLY_ONE 26
#define ONLY_TWO 676
#define ONLY_THREE 17576
#define ONLY_FOUR 456976 
#define ONLY_FIVE 11881376 
#define TOTAL_ONE 26
#define TOTAL_TWO 702
#define TOTAL_THREE 18278
#define TOTAL_FOUR 475254
#define TOTAL_FIVE 12356624


static unsigned qrand_x = 123456789;
static unsigned qrand_y = 678912345;
static unsigned qrand_z = 987651234;

/*
 * 	Seeds the pseudo-randomic number generated with seed.
 * 	@param seed Unsigned number that will seed the generator.
 */
void qrand_seed(unsigned seed);

/*
 * 	Generates a pseudo-randomic number in range [0, UINT_MAX]. The 
 * function is declared here as inline for performance reasons.
 * 	@return Returns a randomic integer in range [0, UINT_MAX].	
 */
static inline unsigned qrand(void)
{
	unsigned t;
	qrand_x ^= qrand_x<<16;
	qrand_x ^= qrand_x>>5;
	qrand_x ^= qrand_x<<1;
	t = qrand_x;
	qrand_x = qrand_y;
	qrand_y = qrand_z;
	qrand_z = t^qrand_x^qrand_y;
	return qrand_z; }

/*  	Unit test to determine if qrand generates a uniform distribution
 * in [0, n] using iterations iterations.
 * 	@param n The superior limit to the range which will be tested.
 * 	@param iterations Number of iterations in the test. 
 */
void qrand_test(unsigned n, unsigned long long iterations);

/*
 *	Sorts a string with 1 to 5 letters.
 * 	@param s Pointer to the char array in which the null-terminated 
 * string will be written to.
 */
void qrand_word(char *s);

#endif
