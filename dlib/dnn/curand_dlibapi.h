// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuRAND_H_
#define DLIB_DNN_CuRAND_H_

#ifdef DLIB_USE_CUDA

#include "tensor.h"
#include "cuda_errors.h"

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        class curand_generator
        {
        public:
            // not copyable
            curand_generator(const curand_generator&) = delete;
            curand_generator& operator=(const curand_generator&) = delete;

            curand_generator() : curand_generator(0) {}
            curand_generator(unsigned long long seed);
            ~curand_generator();

            void fill_gaussian (
                tensor& data,
                float mean = 0,
                float stddev = 1
            );
            /*!
                requires
                    - data.size()%2 == 0
                    - stddev >= 0
                ensures
                    - Fills data with random numbers drawn from a Gaussian distribution
                      with the given mean and standard deviation.
            !*/

            void fill_uniform (
                tensor& data
            );
            /*!
                ensures
                    - Fills data with uniform random numbers in the range (0.0, 1.0].
            !*/

        private:

            void* handle;
        };

    // -----------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuRAND_H_



