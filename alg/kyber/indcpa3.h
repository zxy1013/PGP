
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef INDCPA3_H
#define INDCPA3_H
#include "poly3.h"
#include "polyvec3.h"

#define LARGE_BUFFER_SZ 1024
typedef struct _poly_set4
{
	// Length 1
	poly3* a;
	poly3* b;
	poly3* c;
	poly3* d;

	// Length 4
	polyvec* AV;

	// Length 1
	polyvec* av;
	polyvec* bv;
	polyvec* cv;
	polyvec* dv;
	polyvec* ev;
	polyvec* fv;

	// Length [2 * KYBER_SYMBYTES]
	unsigned char* seed;
	
	// LARGE_BUFFER_SZ = Length in bytes
	unsigned char* large_buffer_a;
	unsigned char* large_buffer_b;
} poly_set4;


void indcpa_keypair(int COUNT, poly_set4* ps, unsigned char *pk,
                    unsigned char *sk, unsigned char* rng_buf, cudaStream_t stream);

void indcpa_enc(int COUNT, poly_set4* ps, unsigned char *c,
                unsigned char *m,
                unsigned char *pk,
                unsigned char *coins, cudaStream_t stream);

void indcpa_dec(int COUNT, poly_set4* ps, unsigned char *m,
                unsigned char *c,
                unsigned char *sk, cudaStream_t stream);

void print_data(const char* text, unsigned char* data, int length);

#endif
