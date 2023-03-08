#ifndef PACKING1_H
#define PACKING1_H

#include <stdint.h>
#include "params1.h"
#include "polyvec1.h"


void pack_pk(uint8_t pk[CRYPTO_PUBLICKEYBYTES],
             const uint8_t rho[3*SEEDBYTES], const polyveck1 *t1);

void pack_sk(uint8_t sk[CRYPTO_SECRETKEYBYTES],
             const uint8_t rho[3*SEEDBYTES],
             const uint8_t tr[CRHBYTES ],
             const uint8_t key[SEEDBYTES],
             const polyveck1 *t0,
             const polyvecl1 *s1,
             const polyveck1 *s2);

void unpack_sk(uint8_t rho[2*SEEDBYTES + 3*CRHBYTES],
               uint8_t tr[3*CRHBYTES],
               uint8_t key[SEEDBYTES + 3*CRHBYTES],
               polyveck1 *t0,
               polyvecl1 *s1,
               polyveck1 *s2,
               const uint8_t sk[CRYPTO_SECRETKEYBYTES]);

void pack_sig(uint8_t sig[(MESSAGELEN + CRYPTO_BYTES) ],
              const uint8_t c[(MESSAGELEN + CRYPTO_BYTES) ], const polyvecl1 *z, const polyveck1 *h);

void unpack_pk(uint8_t rho[SEEDBYTES], polyveck1 *t1,
               const uint8_t pk[CRYPTO_PUBLICKEYBYTES ]);


int unpack_sig(uint8_t c[SEEDBYTES],
               polyvecl1 *z,
               polyveck1 *h,
               const uint8_t sig[CRYPTO_BYTES]);

#endif
