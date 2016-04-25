// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuRAND_CPP_
#define DLIB_DNN_CuRAND_CPP_

#ifdef DLIB_USE_CUDA

#include "curand_dlibapi.h"
#include <curand.h>
#include "../string.h"

static const char* curand_get_error_string(curandStatus_t s)
{
    switch(s)
    {
        case CURAND_STATUS_NOT_INITIALIZED: 
            return "CUDA Runtime API initialization failed.";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "The requested length must be a multiple of two.";
        default:
            return "A call to cuRAND failed";
    }
}

// Check the return value of a call to the cuDNN runtime for an error condition.
#define CHECK_CURAND(call)                                                      \
do{                                                                              \
    const curandStatus_t error = call;                                         \
    if (error != CURAND_STATUS_SUCCESS)                                        \
    {                                                                          \
        std::ostringstream sout;                                               \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
        sout << "code: " << error << ", reason: " << curand_get_error_string(error);\
        throw dlib::curand_error(sout.str());                            \
    }                                                                          \
}while(false)

namespace dlib
{
    namespace cuda 
    {

    // ----------------------------------------------------------------------------------------

        curand_generator::
        curand_generator(
            unsigned long long seed
        ) : handle(nullptr)
        {
            curandGenerator_t gen;
            CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
            handle = gen;

            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
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

            CHECK_CURAND(curandGenerateNormal((curandGenerator_t)handle, 
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

            CHECK_CURAND(curandGenerateUniform((curandGenerator_t)handle, data.device(), data.size()));
        }

    // -----------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuRAND_CPP_

