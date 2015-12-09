// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CUDA_UtILS_H_
#define DLIB_CUDA_UtILS_H_

#ifndef DLIB_USE_CUDA
#error "This file shouldn't be #included unless DLIB_USE_CUDA is #defined"
#endif

#include "cuda_errors.h"

#include <cuda_runtime.h>
#include <sstream>


// Check the return value of a call to the CUDA runtime for an error condition.
#define CHECK_CUDA(call)                                                       \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        std::ostringstream sout;                                               \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
        sout << "code: " << error << ", reason: " << cudaGetErrorString(error);\
        throw dlib::cuda_error(sout.str());                                          \
    }                                                                          \
}

// ----------------------------------------------------------------------------------------

#ifdef __CUDACC__

namespace dlib
{
    namespace cuda
    {

    // ------------------------------------------------------------------------------------

        // This function is from the article:
        // http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
        __inline__ __device__ float warp_reduce_sum(float val) 
        {
            for (int offset = warpSize/2; offset > 0; offset /= 2) 
                val += __shfl_down(val, offset);
            return val;
        }

        __inline__ __device__ bool is_first_thread_in_warp()
        {
            return (threadIdx.x & (warpSize - 1)) == 0;
        }

        __inline__ __device__ void warp_reduce_atomic_add(
            float& out, 
            float val
        ) 
        /*!
            ensures
                - Atomically adds all the val variables in the current warp to out.
                  See this page for an extended discussion: 
                  http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
        !*/
        {
            val = warp_reduce_sum(val);
            if (is_first_thread_in_warp())
                atomicAdd(&out, val);
        }

    // ------------------------------------------------------------------------------------

        class grid_stride_range
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a tool for making a for loop that loops over an entire block of
                    memory inside a kernel, but doing so in a way that parallelizes
                    appropriately across all the threads in a kernel launch.  For example,
                    the following kernel would add the vector a to the vector b and store
                    the output in out (assuming all vectors are of dimension n):
                        __global__ void add_arrays(
                            const float* a, 
                            const float* b, 
                            float* out, 
                            size_t n
                        )
                        {
                            for (auto i : grid_stride_range(0, n))
                            {
                                out[i] = a[i]+b[i];
                            }
                        }
            !*/

        public:
            __device__ grid_stride_range(
                size_t ibegin_,
                size_t iend_
            ) : 
                ibegin(ibegin_),
                iend(iend_)
            {}

            class iterator
            {
            public:
                __device__ iterator() {}
                __device__ iterator(size_t pos_) : pos(pos_) {}

                __device__ size_t operator*() const
                {
                    return pos;
                }

                __device__ iterator& operator++()
                {
                    pos += gridDim.x * blockDim.x;
                    return *this;
                }

                __device__ bool operator!=(const iterator& item) const
                { return pos < item.pos; }

            private:
                size_t pos;
            };

            __device__ iterator begin() const
            {
                return iterator(ibegin+blockDim.x * blockIdx.x + threadIdx.x);
            }
            __device__ iterator end() const
            {
                return iterator(iend);
            }
        private:

            size_t ibegin;
            size_t iend;
        };

    }
}

#endif // __CUDACC__

// ----------------------------------------------------------------------------------------

#endif // DLIB_CUDA_UtILS_H_

