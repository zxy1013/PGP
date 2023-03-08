#ifndef POLY1_H
#define POLY1_H

#include <stdint.h>
#include "params1.h"

typedef struct {
  int32_t coeffs[N];
} poly1;



unsigned int poly_make_hint(poly1 *h, const poly1 *a0, const poly1 *a1);
int poly_chknorm(const poly1 *a, int32_t B);
void poly_uniform(poly1 *a,
                  const uint8_t seed[SEEDBYTES],
                  uint16_t nonce);
void poly_uniform_eta(poly1 *a,
                      const uint8_t seed[SEEDBYTES],
                      uint16_t nonce);
void poly_uniform_gamma1(poly1 *a,
                         const uint8_t seed[CRHBYTES],
                         uint16_t nonce);
void poly_challenge(poly1 *c, const uint8_t seed[MESSAGELEN + CRYPTO_BYTES]);
void poly_challenge1(poly1 *c, const uint8_t seed[SEEDBYTES]);
void polyeta_pack(uint8_t *r, const poly1 *a);
void polyeta_unpack(poly1 *r, const uint8_t *a);
void polyt1_pack(uint8_t *r, const poly1 *a);
void polyt1_unpack(poly1 *r, const uint8_t *a);
void polyt0_pack(uint8_t *r, const poly1 *a);
void polyt0_unpack(poly1 *r, const uint8_t *a);
void polyz_pack(uint8_t *r, const poly1 *a);
void polyz_unpack(poly1 *r, const uint8_t *a);
void polyw1_pack(uint8_t *r, const poly1 *a);

#endif
