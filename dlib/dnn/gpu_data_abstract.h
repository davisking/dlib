// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_GPU_DaTA_ABSTRACT_H_
#ifdef DLIB_GPU_DaTA_ABSTRACT_H_

#include "cuda_errors.h"
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class gpu_data 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a block of size() floats, all stored contiguously in memory.
                Importantly, it keeps two copies of the floats, one on the host CPU side
                and another on the GPU device side. It automatically performs the necessary
                host/device transfers to keep these two copies of the data in sync.

                All transfers to the device happen asynchronously with respect to the
                default CUDA stream so that CUDA kernel computations can overlap with data
                transfers.  However, any transfers from the device to the host happen
                synchronously in the default CUDA stream.  Therefore, you should perform
                all your CUDA kernel launches on the default stream so that transfers back
                to the host do not happen before the relevant computations have completed.

                If DLIB_USE_CUDA is not #defined then this object will not use CUDA at all.
                Instead, it will simply store one host side memory block of floats.  

            THREAD SAFETY
                Instances of this object are not thread-safe.  So don't touch one from
                multiple threads at the same time.
        !*/
    public:

        gpu_data(
        );
        /*!
            ensures
                - #size() == 0
                - #host() == nullptr 
                - #device() == nullptr 
        !*/

        // This object is not copyable, however, it is movable.
        gpu_data(const gpu_data&) = delete;
        gpu_data& operator=(const gpu_data&) = delete;
        gpu_data(gpu_data&& item);
        gpu_data& operator=(gpu_data&& item);


        void async_copy_to_device(
        ); 
        /*!
            ensures
                - This function does not block.
                - if (the host version of the data is newer than the device's copy) then
                    - Begins asynchronously copying host data to the device.
                    - A call to device() that happens before the transfer completes will
                      block until the transfer is complete.  That is, it is safe to call
                      async_copy_to_device() and then immediately call device().
        !*/

        void set_size(
            size_t new_size
        );
        /*!
            ensures
                - #size() == new_size
        !*/

        const float* host(
        ) const;
        /*!
            ensures
                - returns a pointer to the host memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (the host's copy of the data is out of date) then
                    - copies the data from the device to the host, while this is happening
                      the call to host() blocks. 
        !*/

        float* host(
        );
        /*!
            ensures
                - returns a pointer to the host memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (the host's copy of the data is out of date) then
                    - copies the data from the device to the host, while this is happening
                      the call to host() blocks. 
                - Marks the device side data as out of date so that the next call to
                  device() will perform a host to device transfer.  If you want to begin
                  the transfer immediately then you can call async_copy_to_device() after
                  calling host().
        !*/

        const float* device(
        ) const;
        /*!
            requires
                - DLIB_USE_CUDA is #defined
            ensures
                - returns a pointer to the device memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (the device's copy of the data is out of date) then
                    - copies the data from the host to the device, while this is happening
                      the call to device() blocks. 
        !*/

        float* device(
        );
        /*!
            requires
                - DLIB_USE_CUDA is #defined
            ensures
                - returns a pointer to the device memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (the device's copy of the data is out of date) then
                    - copies the data from the host to the device, while this is happening
                      the call to device() blocks. 
                - Marks the host side data as out of date so that the next call to
                  host() will perform a device to host transfer.
        !*/

        size_t size(
        ) const; 
        /*!
            ensures
                - returns the number of floats contained in this object.
        !*/

        void swap (
            gpu_data& item
        );
        /*!
            ensures
                - swaps the state of *this and item
        !*/

    };

    void serialize(const gpu_data& item, std::ostream& out);
    void deserialize(gpu_data& item, std::istream& in);
    /*!
        provides serialization support
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GPU_DaTA_ABSTRACT_H_

