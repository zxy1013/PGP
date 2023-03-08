#ifndef REDUCE2_H
#define REDUCE2_H

#include <stdint.h>
#include "params2.h"

#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32

__device__ int32_t montgomery_reduce(int64_t a);


__device__ int32_t reduce32(int32_t a);

__device__ int32_t caddq(int32_t a);

#endif
