// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuSOLVER_CU_
#define DLIB_DNN_CuSOLVER_CU_

#ifdef DLIB_USE_CUDA

#include "cusolver_dlibapi.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuda_utils.h"

// ----------------------------------------------------------------------------------------

static const char* cusolver_get_error_string(cusolverStatus_t s)
{
    switch(s)
    {
        case CUSOLVER_STATUS_NOT_INITIALIZED: 
            return "CUDA Runtime API initialization failed.";
        case CUSOLVER_STATUS_ALLOC_FAILED: 
            return "CUDA Resources could not be allocated.";
        default:
            return "A call to cuSolver failed";
    }
}

// Check the return value of a call to the cuSolver runtime for an error condition.
#define CHECK_CUSOLVER(call)                                                      \
do{                                                                              \
    const cusolverStatus_t error = call;                                         \
    if (error != CUSOLVER_STATUS_SUCCESS)                                        \
    {                                                                          \
        std::ostringstream sout;                                               \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
        sout << "code: " << error << ", reason: " << cusolver_get_error_string(error);\
        throw dlib::cusolver_error(sout.str());                                \
    }                                                                          \
}while(false)

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        class cusolver_context
        {
        public:
            // not copyable
            cusolver_context(const cusolver_context&) = delete;
            cusolver_context& operator=(const cusolver_context&) = delete;

            cusolver_context()
            {
                handles.resize(16);
            }
            ~cusolver_context()
            {
                for (auto h : handles)
                {
                    if (h)
                        cusolverDnDestroy(h);
                }
            }

            cusolverDnHandle_t get_handle (
            )  
            { 
                int new_device_id;
                CHECK_CUDA(cudaGetDevice(&new_device_id));
                // make room for more devices if needed
                if (new_device_id >= (long)handles.size())
                    handles.resize(new_device_id+16);

                // If we don't have a handle already for this device then make one
                if (!handles[new_device_id])
                    CHECK_CUSOLVER(cusolverDnCreate(&handles[new_device_id]));

                // Finally, return the handle for the current device
                return handles[new_device_id];
            }

        private:

            std::vector<cusolverDnHandle_t> handles;
        };

        static cusolverDnHandle_t context()
        {
            thread_local cusolver_context c;
            return c.get_handle();
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        __global__ void _cuda_set_to_identity_matrix(float* m, size_t nr)
        {
            for (auto j : grid_stride_range(0, nr*nr))
            {
                if (j%(nr+1) == 0)
                    m[j] = 1;
                else
                    m[j] = 0;
            }
        }

        void set_to_identity_matrix (
            tensor& m 
        )
        {
            DLIB_CASSERT(m.size() == m.num_samples()*m.num_samples());
            launch_kernel(_cuda_set_to_identity_matrix, max_jobs(m.size()), m.device(), m.num_samples());
        }

    // ------------------------------------------------------------------------------------

        inv::~inv()
        {
            sync_if_needed();
        }

    // ------------------------------------------------------------------------------------

        void inv::
        operator() (
            const tensor& m_,
            resizable_tensor& out
        )
        {
            DLIB_CASSERT(m_.size() == m_.num_samples()*m_.num_samples(), "Input matrix must be square if you want to invert it.");
            m = m_;

            out.copy_size(m);
            set_to_identity_matrix(out);

            const int nc = m.num_samples();
            int Lwork;
            CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(context(), nc , nc, m.device(), nc, &Lwork));

            if (Lwork > (int)workspace.size())
            {
                sync_if_needed();
                workspace = cuda_data_ptr<float>(Lwork);
            }
            if (nc > (int)Ipiv.size())
            {
                sync_if_needed();
                Ipiv = cuda_data_ptr<int>(nc);
            }
            if (info.size() != 1)
            {
                info = cuda_data_ptr<int>(1);
            }

            CHECK_CUSOLVER(cusolverDnSgetrf(context(), nc, nc, m.device(), nc, workspace, Ipiv, info));
            CHECK_CUSOLVER(cusolverDnSgetrs(context(), CUBLAS_OP_N, nc, nc, m.device(), nc, Ipiv, out.device(), nc, info));
            did_work_lately = true;
        }

    // ------------------------------------------------------------------------------------

        int inv::
        get_last_status(
        )
        {
            std::vector<int> linfo; 
            memcpy(linfo, info);
            if (linfo.size() != 0)
                return linfo[0];
            else
                return 0;
        }

    // ------------------------------------------------------------------------------------

        void inv::
        sync_if_needed()
        {
            if (did_work_lately)
            {
                did_work_lately = false;
                // make sure we wait until any previous kernel launches have finished
                // before we do something like deallocate the GPU memory.
                cudaDeviceSynchronize();
            }
        }

    // ------------------------------------------------------------------------------------

    }  
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuSOLVER_CU_


