#ifndef ROUNDING2_H
#define ROUNDING2_H

#include <stdint.h>
#include "params2.h"

__device__ int32_t power2round(int32_t *a0, int32_t a);



__device__ int32_t decompose(int32_t *a0, int32_t a);


__device__ unsigned int make_hint(int32_t a0, int32_t a1);



__device__ int32_t use_hint(int32_t a, unsigned int hint);

#endif
