#ifndef ROUNDING1_H
#define ROUNDING1_H

#include <stdint.h>
#include "params1.h"


__device__ int32_t power2round1(int32_t *a0, int32_t a);
__device__ int32_t decompose1(int32_t *a0, int32_t a);
__device__ int32_t use_hint1(int32_t a, unsigned int hint);


unsigned int make_hint1(int32_t a0, int32_t a1);
#endif
