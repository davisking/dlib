// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cuda_utils.h"
#include "cuda_dlib.h"


// ------------------------------------------------------------------------------------

__global__ void cuda_add_arrays(const float* a, const float* b, float* out, size_t n)
{
   out[0] += a[0]+b[0];
}

void add_arrays()
{
   cuda_add_arrays<<<512,512>>>(0,0,0,0);
}

// ------------------------------------------------------------------------------------

