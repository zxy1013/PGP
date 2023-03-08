#ifndef NTT2_H
#define NTT2_H

#include <stdint.h>
#include "params2.h"

__device__ void ntt(int32_t a[N]);

__device__ void invntt_tomont(int32_t a[N]);

#endif
