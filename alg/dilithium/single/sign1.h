#ifndef SIGN1_H
#define SIGN1_H

#include <stddef.h>
#include <stdint.h>
#include "params1.h"
#include "polyvec1.h"
#include "poly1.h"
#include "fips2021.h"

typedef struct {
    polyvecl1* z; 
    polyvecl1* zh;
    polyvecl1* mat;
    polyvecl1* math;
    polyvecl1* s1h;
    polyveck1* t0; 
    polyveck1* t0h;
    polyveck1* s2;
    polyveck1* s2h;
    polyveck1* w0;
    polyveck1* w0h;
    uint8_t* muh;
} sign_init1;

typedef struct {
    polyvecl1* z;
    polyvecl1* zh;
    polyvecl1* mat;
    polyvecl1* math;
    polyvecl1* s1;
    polyvecl1* s1h;
    polyvecl1* y;
    polyvecl1* yh;
    polyveck1* w1;
    polyveck1* w1h;
    polyveck1* t0; 
    polyveck1* t0h;
    polyveck1* s2;
    polyveck1* s2h;
    polyveck1* w0;
    polyveck1* w0h;
    poly1* cp;
    poly1* cph;
    polyveck1* h;
    polyveck1* hh;
    keccak_state1* stateh;
} sign_sig1;



typedef struct {
    polyvecl1* z;
    polyvecl1* zh;
    polyvecl1* mat;
    polyvecl1* math;
    polyveck1* w1;
    polyveck1* w0;
    polyveck1* w0h;
    poly1* cp;
    poly1* cph;
    uint8_t* muh;
    polyveck1* h;
    polyveck1* hh;
    uint8_t* buf;
    uint8_t* bufh;
    uint8_t* c2h;
    uint8_t* ch;
    keccak_state1* stateh;
    uint8_t* rhoh;
} sign_verify1;


int crypto_sign_keypair(uint8_t *seedbuf, sign_init1* keypair,uint8_t *pk, uint8_t *sk);



int crypto_sign_sig1nature(uint8_t *seedbuf, sign_sig1* sign,uint8_t *sig, size_t *siglen,
                          const uint8_t *m, size_t mlen,
                          const uint8_t *sk);


int crypto_sign(uint8_t *seedbuf, sign_sig1* sign, uint8_t *sm, size_t *smlen,
                const uint8_t *m, size_t mlen,
                const uint8_t *sk);


int crypto_sign_verify(sign_verify1* verify, const uint8_t *sig, size_t siglen,
                       const uint8_t *m, size_t mlen,
                       const uint8_t *pk);

int crypto_sign_open(sign_verify1* verify, uint8_t *m, size_t *mlen,
                     const uint8_t *sm, size_t smlen,
                     const uint8_t *pk);

#endif
