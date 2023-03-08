
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef API3_H
#define API3_H

#include "params3.h"

#define CRYPTO_SECRETKEYBYTES1  KYBER_SECRETKEYBYTES
#define CRYPTO_PUBLICKEYBYTES1  KYBER_PUBLICKEYBYTES
#define CRYPTO_CIPHERTEXTBYTES KYBER_CIPHERTEXTBYTES
#define CRYPTO_BYTES1           KYBER_SSBYTES

#define CRYPTO_ALGNAME "Kyber1024"

int crypto_kem_keypair(unsigned char *pk, unsigned char *sk);

int crypto_kem_enc(unsigned char *ct, unsigned char *ss, unsigned char *pk);

int crypto_kem_dec(unsigned char *ss, unsigned char *ct, unsigned char *sk);


#endif