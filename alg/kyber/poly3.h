
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef POLY3_H
#define POLY3_H

#include <stdint.h>
#include "params3.h"

typedef struct {
	int16_t threads[N_TESTS];
} threads;

/*
 * Elements of R_q = Z_q[X]/(X^n + 1). Represents polynomial
 * coeffs[0] + X*coeffs[1] + X^2*xoeffs[2] + ... + X^{n-1}*coeffs[n-1]
 */
typedef struct{
	threads coeffs[KYBER_N];
} poly3;

__device__ void poly_compress(unsigned char *r, poly3 *a);
__device__ void poly_decompress(poly3 *r, unsigned char *a);

__device__ void poly_tobytes(unsigned char *r, poly3 *a);
__device__ void poly_frombytes(poly3 *r, unsigned char *a);

__global__  void poly_frommsg_n(int COUNT, poly3* r,  unsigned char* msg);
__global__  void poly_tomsg_n(int COUNT, unsigned char* msg, poly3* a);

__device__ void poly_frommsg(poly3* r,  unsigned char* msg);
__device__ void poly_tomsg(unsigned char* msg, poly3* a);

__global__ void poly_getnoise(int COUNT, poly3 *r, unsigned char *seed, unsigned char nonce);

__device__ void poly_ntt1(poly3 *r);
__global__ void poly_ntt_n(int COUNT, poly3* r);
__device__ void poly_invntt(poly3 *r);
__global__ void poly_invntt_n(int COUNT, poly3* r);

__device__ void poly_basemul(poly3 *r,  poly3 *a, poly3 *b);
__global__ void poly_frommont_n(int COUNT, poly3 *r);

__global__ void poly_reduce_n(int COUNT, poly3 *r);
__device__ void poly_reduce1(poly3* r);

__device__ void poly_csubq(poly3 *r);

__global__  void poly_add_n(int COUNT, poly3 *r,  poly3 *a,  poly3 *b);
__global__  void poly_sub_n(int COUNT, poly3 *r,  poly3 *a,  poly3 *b);

__device__  void poly_add(poly3* r,  poly3* a,  poly3* b);
__device__  void poly_sub(poly3* r,  poly3* a,  poly3* b);

#endif
