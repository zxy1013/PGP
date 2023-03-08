#ifndef SIGN2_H
#define SIGN2_H

#include <stddef.h>
#include <stdint.h>
#include "params2.h"
#include "polyvec2.h"
#include "poly2.h"
#include "fips2022.h"

typedef struct {
    polyvecl* z; 
    polyvecl* mat;
    polyvecl* s1;
    polyvecl* y;
    polyveck* h;
    polyveck* w1; 
    polyveck* t0; 
    polyveck* s2;
    polyveck* w0; 
    poly* cp;
    keccak_state* state;
    int* flag;
    unsigned int* n;
} sign_sig;

typedef struct {
    polyvecl* z;
    polyvecl* mat;
    polyveck* h;
    polyveck* w1; 
    polyveck* w0;
    uint8_t* rho;
    uint8_t* c;
    uint8_t* buf;
    uint8_t* c2;
    uint8_t* mu;
    poly* cp;
    keccak_state* state;
    int* flag;
    unsigned int* n;
} sign_verify;

typedef struct {
    polyvecl* z; // keypaire s1hat sign z
    polyvecl* mat;
    polyvecl* s1;
    polyvecl* y;
    
    polyveck* h;
    polyveck* w1; 
    polyveck* t0; 
    polyveck* s2;
    polyveck* w0; // keypaire/verify t1 sign w0
    
    uint8_t* rho;
    uint8_t* c;
    uint8_t* buf;
    uint8_t* c2;
    uint8_t* mu; // keypaire tr verify mu 
    
    poly* cp;
    
    keccak_state* state;
    int* flag;
    unsigned int* n;
} sign_sign;

int crypto_sign_keypair(int group,uint8_t *seedbuf, sign_sign* keypair,uint8_t *pk, uint8_t *sk);



int crypto_sign_signature(int group,uint8_t *seedbuf, sign_sig* sign,uint8_t *sig, size_t *siglen,
                          const uint8_t *m, size_t mlen,
                          const uint8_t *sk);


int crypto_sign(int group,uint8_t *seedbuf, sign_sig* sign, uint8_t *sm, size_t *smlen,
                const uint8_t *m, size_t mlen,
                const uint8_t *sk);


void crypto_sign_verify(int group,int* result,sign_verify* verify, const uint8_t *sig, size_t siglen,
                       const uint8_t *m, size_t mlen,
                       const uint8_t *pk);

int crypto_sign_open(int group,sign_verify* verify, uint8_t *m, size_t *mlen,
                     const uint8_t *sm, size_t smlen,
                     const uint8_t *pk);

#endif
