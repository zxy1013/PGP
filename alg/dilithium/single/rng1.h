#ifndef rng1_h
#define rng1_h
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h> 


void rrandombytes11(uint8_t *out, size_t outlen, size_t len, int id);
int randombytes11(unsigned char *x, unsigned long long xlen);

#endif
