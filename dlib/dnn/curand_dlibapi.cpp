// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuRAND_CPP_
#define DLIB_DNN_CuRAND_CPP_

#ifdef DLIB_USE_CUDA

#include "curand_dlibapi.h"
#include <curand.h>
#include "../string.h"

namespace dlib
{
    namespace cuda 
    {

    // ----------------------------------------------------------------------------------------

        // TODO, make into a macro that prints more information like the line number, etc.
        static void check(curandStatus_t s)
        {
            switch(s)
            {
                case CURAND_STATUS_SUCCESS: return;
                case CURAND_STATUS_NOT_INITIALIZED: 
                    throw curand_error("CUDA Runtime API initialization failed.");
                case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                    throw curand_error("The requested length must be a multiple of two.");
                default:
                    throw curand_error("A call to cuRAND failed: " + cast_to_string(s));
            }
        }

    // ----------------------------------------------------------------------------------------

        curand_generator::
        curand_generator(
            unsigned long long seed
        ) : handle(nullptr)
        {
            curandGenerator_t gen;
            check(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
            handle = gen;

            check(curandSetPseudoRandomGeneratorSeed(gen, seed));
        }

        curand_generator::
        ~curand_generator()
        {
            if (handle)
            {
                curandDestroyGenerator((curandGenerator_t)handle);
            }
        }

        void curand_generator::
        fill_gaussian (
            tensor& data,
            float mean,
            float stddev
        )
        {
            if (data.size() == 0)
                return;

            check(curandGenerateNormal((curandGenerator_t)handle, 
                                        data.device(),
                                        data.size(),
                                        mean,
                                        stddev));
        }

        void curand_generator::
        fill_uniform (
            tensor& data
        )
        {
            if (data.size() == 0)
                return;

            check(curandGenerateUniform((curandGenerator_t)handle, data.device(), data.size()));
        }

    // -----------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuRAND_CPP_

