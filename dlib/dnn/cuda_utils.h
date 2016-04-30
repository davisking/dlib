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
do{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        std::ostringstream sout;                                               \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
        sout << "code: " << error << ", reason: " << cudaGetErrorString(error);\
        throw dlib::cuda_error(sout.str());                                          \
    }                                                                          \
}while(false)

// ----------------------------------------------------------------------------------------

#ifdef __CUDACC__

namespace dlib
{
    namespace cuda
    {

    // ------------------------------------------------------------------------------------

        __inline__ __device__ size_t pack_idx (
            size_t dim_size3,
            size_t dim_size2,
            size_t dim_size1,
            size_t idx4,
            size_t idx3,
            size_t idx2,
            size_t idx1
        )
        /*!
            ensures
                - Converts a 4D array index into a 1D index assuming row major layout.  To
                  understand precisely what this function does, imagine we had an array
                  declared like this:
                    int ARRAY[anything][dim_size3][dim_size2][dim_size1];
                  Then we could index it like this:
                    ARRAY[idx4][idx3][idx2][idx1]
                  or equivalently like this:
                    ((int*)ARRAY)[pack_idx(dim_size3,dim_size2,dim_size1, idx4,idx3,idx2,idx1)]
        !*/
        {
            return ((idx4*dim_size3 + idx3)*dim_size2 + idx2)*dim_size1 + idx1;
        }

        __inline__ __device__ void unpack_idx (
            size_t idx,
            size_t dim_size3,
            size_t dim_size2,
            size_t dim_size1,
            size_t& idx4,
            size_t& idx3,
            size_t& idx2,
            size_t& idx1
        )
        /*!
            ensures
                - This function computes the inverse of pack_idx().  Therefore, 
                  if PACKED == pack_idx(dim_size3,dim_size2,dim_size1, idx4,idx3,idx2,idx1)
                  then unpack_idx(PACKED,dim_size3,dim_size2,dim_size1, IDX4,IDX3,IDX2,IDX1)
                  results in:
                    - IDX1 == idx1
                    - IDX2 == idx2
                    - IDX3 == idx3
                    - IDX4 == idx4
        !*/
        {
            idx1 = idx%dim_size1;

            idx /= dim_size1;
            idx2 = idx%dim_size2;

            idx /= dim_size2;
            idx3 = idx%dim_size3;

            idx /= dim_size3;
            idx4 = idx;
        }

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

        struct max_jobs
        {
            max_jobs(size_t n) : num(n) {}
            size_t num;
        };

        template <typename Kernel, typename... T>
        void launch_kernel (
            Kernel K,
            T ...args
        )
        /*!
            ensures
                - launches the given kernel K(args...).  The point of this function is to
                  automatically set the kernel launch parameters to something reasonable
                  based on the properties of the kernel and the current GPU card.
        !*/
        {
            int num_blocks, num_threads;
            CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&num_blocks,&num_threads,K));
            K<<<num_blocks,num_threads>>>(args...);
        }

        template <typename Kernel, typename... T>
        void launch_kernel (
            Kernel K,
            max_jobs m,
            T ...args
        )
        /*!
            ensures
                - This function is just like launch_kernel(K,args...) except that you can
                  additionally supply a max_jobs number that tells it how many possible
                  total threads could be used.  This is useful when launching potentially
                  small jobs that might not need the number of threads suggested by
                  launch_kernel().  
        !*/
        {
            if (m.num == 0)
                return;
            int num_blocks, num_threads;
            CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&num_blocks,&num_threads,K));
            // Check if the job is really small and we don't really need to launch a kernel
            // with this many blocks and threads.
            if (num_blocks*num_threads > m.num)
                num_blocks = (m.num+num_threads-1)/num_threads;

            K<<<num_blocks,num_threads>>>(args...);
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

    // ------------------------------------------------------------------------------------

        class grid_stride_range_y
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is just like grid_stride_range except that it looks at
                    CUDA's y thread index (e.g. threadIdx.y) instead of the x index.
                    Therefore, if you launch a cuda kernel with a statement like:
                        dim3 blocks(10,1);
                        dim3 threads(32,32);  // You need to have x any not equal to 1 to get parallelism over both loops.
                        add_arrays<<<blocks,threads>>>(a,b,out,nr,nc);
                    You can perform a nested 2D parallel for loop rather than doing just a
                    1D for loop.
                   
                    So the code in the kernel would look like this if you wanted to add two
                    2D matrices:
                        __global__ void add_arrays(
                            const float* a, 
                            const float* b, 
                            float* out, 
                            size_t nr,
                            size_t nc
                        )
                        {
                            for (auto r : grid_stride_range_y(0, nr))
                            {
                                for (auto c : grid_stride_range(0, nc))
                                {
                                    auto i = r*nc+c;
                                    out[i] = a[i]+b[i];
                                }
                            }
                        }
            !*/

        public:
            __device__ grid_stride_range_y(
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
                    pos += gridDim.y * blockDim.y;
                    return *this;
                }

                __device__ bool operator!=(const iterator& item) const
                { return pos < item.pos; }

            private:
                size_t pos;
            };

            __device__ iterator begin() const
            {
                return iterator(ibegin+blockDim.y * blockIdx.y + threadIdx.y);
            }
            __device__ iterator end() const
            {
                return iterator(iend);
            }
        private:

            size_t ibegin;
            size_t iend;
        };

    // ------------------------------------------------------------------------------------

    }
}

#endif // __CUDACC__

// ----------------------------------------------------------------------------------------

#endif // DLIB_CUDA_UtILS_H_

