#ifndef POLYVEC1_H
#define POLYVEC1_H

#include <stdint.h>
#include "params1.h"
#include "poly1.h"

/* Vectors of polynomials of length L */
typedef struct {
  poly1 vec[L];
} polyvecl1;
/* Vectors of polynomials of length K */
typedef struct {
  poly1 vec[K];
} polyveck1;


void polyvecl_uniform_eta(polyvecl1 *Cv, polyvecl1 *v, const uint8_t seed[2*SEEDBYTES], uint16_t nonce);
void polyvecl_uniform_gamma1(polyvecl1 *v, polyvecl1 *Cv,const uint8_t seed[CRHBYTES], uint16_t nonce);
void polyvec_matrix_expand(polyvecl1 mat[K], const uint8_t rho[3*SEEDBYTES]);
void polyvec_matrix_expand1(polyvecl1 mat[K], const uint8_t rho[2*SEEDBYTES + 3*CRHBYTES]);
void polyvec_matrix_expand2(polyvecl1 mat[K], const uint8_t rho[SEEDBYTES]);
void polyveck_uniform_eta(polyveck1 *v, const uint8_t seed[2*SEEDBYTES], uint16_t nonce);
void polyveck_pack_w(uint8_t r[MESSAGELEN + CRYPTO_BYTES], const polyveck1 *w1);
int polyvecl_chknorm(const polyvecl1 *v, int32_t bound);
int polyveck_chknorm(const polyveck1 *v, int32_t bound);
unsigned int polyveck_make_hint(polyveck1 *h,
                                const polyveck1 *v0,
                                const polyveck1 *v1);





__global__ void Gpolyveck_use_hint(polyveck1 *w, const polyveck1 *v, const polyveck1 *h);
__global__ void Gpolyveck_shiftl(polyveck1 *v);
__global__ void Gpolyvecl_reduce(polyvecl1 *v);
__global__ void Gpolyveck_reduce(polyveck1 *v);
__global__ void Gpolyveck_caddq(polyveck1 *v);
__global__ void GpolyK_add(polyveck1 *w, const polyveck1 *u, const polyveck1 *v);
__global__ void GpolyL_add(polyvecl1 *w, const polyvecl1 *u, const polyvecl1 *v);
__global__ void Gpolyveck_sub(polyveck1 *w, const polyveck1 *u, const polyveck1 *v);
__global__ void Gpolyveck_pointwise_poly_montgomery(polyveck1 *r, const poly1 *a, const polyveck1 *v);
__global__ void Gpolyvecl_pointwise_poly_montgomery(polyvecl1 *r, const poly1 *a, const polyvecl1 *v);
__global__ void Gpolyvec_matrix_pointwise_montgomery(polyveck1 *t, const polyvecl1 mat[K], const polyvecl1 *v);
__global__ void Gpolyveck_power2round(polyveck1 *v1, polyveck1 *v0, const polyveck1 *v);
__global__ void Gpolyveck_decompose(polyveck1 *v1, polyveck1 *v0, const polyveck1 *v);
__global__ void Gpolyveck_pack_w1(uint8_t r[K * POLYW1_PACKEDBYTES], const polyveck1 *w1);


#endif
