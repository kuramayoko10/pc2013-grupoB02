#ifndef QRAND_H
#define QRAND_H


void qrand_seed(unsigned i);
unsigned long qrand(void);
void qrand_test(unsigned n, unsigned long long iterations);

#endif
