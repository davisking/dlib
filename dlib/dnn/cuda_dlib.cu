// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cuda_utils.h"
#include "cuda_dlib.h"

namespace dlib 
{ 
    namespace cuda 
    {

    // ------------------------------------------------------------------------------------

        __global__ void cuda_add_arrays(const float* a, const float* b, float* out, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                out[i] += a[i]+b[i];
            }
        }

        void add_arrays(const gpu_data& a, const gpu_data& b, gpu_data& out)
        {
            DLIB_CASSERT(a.size() == b.size(),"");
            out.set_size(a.size());
            cuda_add_arrays<<<512,512>>>(a.device(), b.device(), out.device(), a.size());
        }

    // ------------------------------------------------------------------------------------

    }
}

