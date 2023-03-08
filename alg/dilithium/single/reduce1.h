#ifndef REDUCE1_H
#define REDUCE1_H

#include <stdint.h>
#include "params1.h"

#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32

__device__ int32_t montgomery_reduce1(int64_t a);
__device__ int32_t reduce321(int32_t a);
__device__ int32_t caddq1(int32_t a);

#endif
