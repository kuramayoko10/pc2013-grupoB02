#ifndef QRAND_H
#define QRAND_H


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
	return qrand_z;
}

/*  	Unit test to determine if qrand generates a uniform distribution
 * in [0, n] using iterations iterations.
 * 	@param n The superior limit to the range which will be tested.
 * 	@param iterations Number of iterations in the test. 
 */
void qrand_test(unsigned n, unsigned long long iterations);
void qrand_word(char *, unsigned);

#endif
