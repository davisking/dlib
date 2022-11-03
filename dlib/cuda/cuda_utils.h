// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CUDA_UtILS_H_
#define DLIB_CUDA_UtILS_H_

#include "../algs.h"

#ifndef DLIB_USE_CUDA
#error "This file shouldn't be #included unless DLIB_USE_CUDA is #defined"
#endif

#include "cuda_errors.h"
#include <cmath>

#include <cuda_runtime.h>
#include <sstream>
#include <iostream>
#include <memory>
#include <vector>
#include <type_traits>


// Check the return value of a call to the CUDA runtime for an error condition.
#define CHECK_CUDA(call)                                                       \
do{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        std::ostringstream sout;                                               \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
        sout << "code: " << cudaGetLastError() << ", reason: " << cudaGetErrorString(error);\
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
#if CUDART_VERSION >= 9000
                val += __shfl_down_sync(0xFFFFFFFF,val, offset);
#else
                val += __shfl_down(val, offset);
#endif
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
            max_jobs(int x) : num_x(x) {}
            max_jobs(int x, int y) : num_x(x), num_y(y) {}
            int num_x;
            int num_y = 1;
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
            if (m.num_x == 0 || m.num_y == 0)
                return;
            int num_blocks, num_threads;
            CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&num_blocks,&num_threads,K));
            // Check if the job is really small and we don't really need to launch a kernel
            // with this many blocks and threads.
            if (num_blocks*num_threads > m.num_x*m.num_y)
                num_blocks = (m.num_x*m.num_y+num_threads-1)/num_threads;

            if (m.num_y == 1)
            {
                K<<<num_blocks,num_threads>>>(args...);
            }
            else
            {
                /*
                    In general, the reason m.num_y!=1 (i.e. the reason you are in this
                    code path) is because we are using nested grid-stride loops.  There are
                    two important things to note about what we are doing here.  To
                    illustrate them we will talk about this little CUDA code snippet:

                        // initialize out before we begin.
                        for (auto i : grid_stride_range_y(0, nr))
                            for (auto j : grid_stride_range(0, 1))
                                out[i] = 0;

                        __syncthreads(); // synchronize threads in block

                        // loop over some 2D thing and sum and store things into out.
                        for (auto i : grid_stride_range_y(0, nr))
                        {
                            float temp = 0;
                            for (auto j : grid_stride_range(0, nc))
                                temp += whatever[i*nc+j];

                            // store the sum into out[i]
                            warp_reduce_atomic_add(out[i], temp);
                        }
                    
                    First, we make sure the number of x threads is a multiple of 32 so that
                    you can use warp_reduce_atomic_add() inside the y loop.  
                    
                    Second, we put the x block size to 1 so inter-block synchronization is
                    easier.  For example, if the number of x blocks wasn't 1 the above code
                    would have a race condition in it.  This is because the execution of
                    out[i]=0 would be done by blocks with blockIdx.x==0, but then in the
                    second set of loops, *all* the x blocks use out[i].  Since
                    __syncthreads() doesn't do any synchronization between blocks some of
                    the blocks might begin before the out[i]=0 statements finished and that
                    would be super bad.
                */
                
                // Try and make sure that the ratio of x to y threads is reasonable based
                // on the respective size of our loops.
                int x_threads = 32;
                int y_threads = num_threads/32;
                const int ratio = static_cast<int>(std::round(put_in_range(1, y_threads, m.num_x/(double)m.num_y)));
                x_threads *= ratio;
                y_threads /= ratio;

                dim3 blocks(1,num_blocks);  
                dim3 threads(x_threads,y_threads); 
                K<<<blocks,threads>>>(args...);
            }
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
                        dim3 blocks(1,10);
                        dim3 threads(32,32);  // You need to have x and y not equal to 1 to get parallelism over both loops.
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

