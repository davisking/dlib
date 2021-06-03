// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuSOLVER_H_
#define DLIB_DNN_CuSOLVER_H_

#ifdef DLIB_USE_CUDA

#include "tensor.h"
#include "cuda_errors.h"
#include "cuda_data_ptr.h"
#include "../noncopyable.h"

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        class inv : noncopyable
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a functor for doing matrix inversion on the GPU.  The only
                    reason it's an object is to avoid the reallocation of some GPU memory
                    blocks if you want to do a bunch of matrix inversions in a row.
            !*/

        public:

            inv() = default;
            ~inv();

            void operator() (
                const tensor& m,
                resizable_tensor& out
            );
            /*!
                requires
                    - m.size() == m.num_samples()*m.num_samples()
                      (i.e. mat(m) must be a square matrix)
                ensures
                    - out == inv(mat(m));
            !*/

            int get_last_status(
            );
            /*!
                ensures
                    - returns 0 if the last matrix inversion was successful and != 0
                      otherwise.
            !*/

        private:

            void sync_if_needed();

            bool did_work_lately = false;
            resizable_tensor m;
            cuda_data_ptr<float> workspace;
            cuda_data_ptr<int> Ipiv;
            cuda_data_ptr<int> info;
        };

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuSOLVER_H_



