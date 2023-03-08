#ifndef SYMMETRIC2_H
#define SYMMETRIC2_H

#include <stdint.h>
#include "params2.h"


#include "fips2022.h"

typedef keccak_state stream128_state;
typedef keccak_state stream256_state;


__device__ void dilithium_shake128_stream_init(keccak_state *state,
                                    const uint8_t seed[SEEDBYTES],
                                    uint16_t nonce);


__device__ void dilithium_shake256_stream_init(keccak_state *state,
                                    const uint8_t seed[CRHBYTES],
                                    uint16_t nonce);


#define STREAM128_BLOCKBYTES SHAKE128_RATE
#define STREAM256_BLOCKBYTES SHAKE256_RATE


#define stream128_init(STATE, SEED, NONCE) \
        dilithium_shake128_stream_init(STATE, SEED, NONCE)
#define stream128_squeezeblocks(OUT, OUTBLOCKS, STATE) \
        shake128_squeezeblocks(OUT, OUTBLOCKS, STATE)

#define stream256_init(STATE, SEED, NONCE) \
        dilithium_shake256_stream_init(STATE, SEED, NONCE)
#define stream256_squeezeblocks(OUT, OUTBLOCKS, STATE) \
        shake256_squeezeblocks(OUT, OUTBLOCKS, STATE)


#endif
