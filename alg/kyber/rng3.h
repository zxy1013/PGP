//
//  rng.h
//
//  Created by Bassham, Lawrence E (Fed) on 8/29/17.
//  Copyright © 2017 Bassham, Lawrence E (Fed). All rights reserved.
//
//  @Author: Arpan Jati
//  Adapted from NewHope Reference Codebase and Parallelized using CUDA// 
//  Modified to generate constant output for debugging. Do not use in actual application.
//  Updated: August 2019

#ifndef rng3_h
#define rng3_h
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define RNG_SUCCESS      0
#define RNG_BAD_MAXLEN  -1
#define RNG_BAD_OUTBUF  -2
#define RNG_BAD_REQ_LEN -3

void randombytes_init(unsigned char* entropy_input,
	unsigned char* personalization_string,
	int security_strength);
void rrandombytes(uint8_t *out, size_t outlen, int id);
int randombytes1(unsigned char* x, unsigned long long xlen);

__device__ int randombytes_device(unsigned char* x, unsigned long long xlen);

#endif /* rng_h */
