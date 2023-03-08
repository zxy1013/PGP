#include "params1.h"
#include "packing1.h"
#include "polyvec1.h"
#include "poly1.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


void pack_pk(uint8_t pk[CRYPTO_PUBLICKEYBYTES],
             const uint8_t rho[SEEDBYTES],
             const polyveck1 *t1)
{
  unsigned int i;

  for(i = 0; i < SEEDBYTES; ++i)
    pk[i] = rho[i];
  pk += SEEDBYTES;
  for(i = 0; i < K; ++i)
    polyt1_pack(pk + i*POLYT1_PACKEDBYTES, &t1->vec[i]);
}



void pack_sk(uint8_t sk[CRYPTO_SECRETKEYBYTES],
             const uint8_t rho[SEEDBYTES],
             const uint8_t tr[CRHBYTES],
             const uint8_t key[SEEDBYTES],
             const polyveck1 *t0,
             const polyvecl1 *s1,
             const polyveck1 *s2)
{
  unsigned int i;

  for(i = 0; i < SEEDBYTES; ++i)
    sk[i] = rho[i];
  sk += SEEDBYTES;
    
  for(i = 0; i < SEEDBYTES; ++i)
    sk[i] = key[i];
  sk += SEEDBYTES;

  for(i = 0; i < CRHBYTES; ++i)
    sk[i] = tr[i];
  sk += CRHBYTES;
    
  for(i = 0; i < L; ++i)
    polyeta_pack(sk + i*POLYETA_PACKEDBYTES, &s1->vec[i]);
  sk += L*POLYETA_PACKEDBYTES;

  for(i = 0; i < K; ++i)
    polyeta_pack(sk + i*POLYETA_PACKEDBYTES, &s2->vec[i]);
  sk += K*POLYETA_PACKEDBYTES;
    
  for(i = 0; i < K; ++i)
    polyt0_pack(sk + i*POLYT0_PACKEDBYTES, &t0->vec[i]);
}



void unpack_sk(uint8_t rho[SEEDBYTES],
               uint8_t tr[CRHBYTES],
               uint8_t key[SEEDBYTES],
               polyveck1 *t0,
               polyvecl1 *s1,
               polyveck1 *s2,
               const uint8_t sk[CRYPTO_SECRETKEYBYTES])
{
  unsigned int i;

  for(i = 0; i < SEEDBYTES; ++i)
    rho[i] = sk[i];
  sk += SEEDBYTES;

  for(i = 0; i < SEEDBYTES; ++i)
    key[i] = sk[i];
  sk += SEEDBYTES;
  for(i = 0; i < CRHBYTES; ++i)
    tr[i] = sk[i];
  sk += CRHBYTES;
  for(i=0; i < L; ++i)
    polyeta_unpack(&s1->vec[i], sk + i*POLYETA_PACKEDBYTES);
  sk += L*POLYETA_PACKEDBYTES;

  for(i=0; i < K; ++i)
    polyeta_unpack(&s2->vec[i], sk + i*POLYETA_PACKEDBYTES);
  sk += K*POLYETA_PACKEDBYTES;

  for(i=0; i < K; ++i)
    polyt0_unpack(&t0->vec[i], sk + i*POLYT0_PACKEDBYTES);
}



void unpack_pk(uint8_t rho[SEEDBYTES],
               polyveck1 *t1,
               const uint8_t pk[CRYPTO_PUBLICKEYBYTES])
{
  unsigned int i;

  for(i = 0; i < SEEDBYTES; ++i)
    rho[i] = pk[i];
  pk += SEEDBYTES;

  for(i = 0; i < K; ++i)
    polyt1_unpack(&t1->vec[i], pk + i*POLYT1_PACKEDBYTES);
}



void pack_sig(uint8_t sig[(MESSAGELEN + CRYPTO_BYTES)],
              const uint8_t c[MESSAGELEN + CRYPTO_BYTES], const polyvecl1 *z, const polyveck1 *h){
  unsigned int i, j, k;

  for(i=0; i < SEEDBYTES; ++i)
    sig[i] = c[i];

  for(i = 0; i < L; ++i)
    polyz_pack(sig + i*POLYZ_PACKEDBYTES + SEEDBYTES, &z->vec[i]);

  // Encode h 
  for(i = 0; i < OMEGA + K; ++i)
    sig[i+ SEEDBYTES+L*POLYZ_PACKEDBYTES] = 0;

  k = 0;
  for(i = 0; i < K; ++i) {
    for(j = 0; j < N; ++j)
      if(h->vec[i].coeffs[j] != 0){
        sig[k+ SEEDBYTES+L*POLYZ_PACKEDBYTES] = j;
        k++;
      }
    sig[OMEGA + i + SEEDBYTES+L*POLYZ_PACKEDBYTES] = k;
  }
}



int unpack_sig(uint8_t c[SEEDBYTES],
               polyvecl1 *z,
               polyveck1 *h,
               const uint8_t sig[CRYPTO_BYTES])
{
  unsigned int i, j, k;

  for(i = 0; i < SEEDBYTES; ++i)
    c[i] = sig[i];
  sig += SEEDBYTES;

  for(i = 0; i < L; ++i)
    polyz_unpack(&z->vec[i], sig + i*POLYZ_PACKEDBYTES);
  sig += L*POLYZ_PACKEDBYTES;

  k = 0;
  for(i = 0; i < K; ++i) {
    for(j = 0; j < N; ++j)
      h->vec[i].coeffs[j] = 0;

    if(sig[OMEGA + i] < k || sig[OMEGA + i] > OMEGA)
      return 1;

    for(j = k; j < sig[OMEGA + i]; ++j) {
      /* Coefficients are ordered for strong unforgeability */
      if(j > k && sig[j] <= sig[j-1]) return 1;
      h->vec[i].coeffs[sig[j]] = 1;
    }

    k = sig[OMEGA + i];
  }

  /* Extra indices are zero for strong unforgeability */
  for(j = k; j < OMEGA; ++j)
    if(sig[j])
      return 1;

  return 0;
}
