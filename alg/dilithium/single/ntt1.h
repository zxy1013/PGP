#ifndef NTT1_H
#define NTT1_H

#include <stdint.h>
#include "params1.h"

__global__ void GNTT(int32_t* a);
__global__ void GINTT(int32_t* a);

#endif
