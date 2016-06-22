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
                - #host_ready() == true
                - #device_ready() == true
                - #device_id() == 0
        !*/

        // This object is not copyable, however, it is movable.
        gpu_data(const gpu_data&) = delete;
        gpu_data& operator=(const gpu_data&) = delete;
        gpu_data(gpu_data&& item);
        gpu_data& operator=(gpu_data&& item);

        int device_id(
        ) const; 
        /*!
            ensures
                - returns the ID of the CUDA device that allocated this memory. I.e. the
                  number returned by cudaGetDevice() when the memory was allocated.
                - If CUDA is not being used then this function always returns 0.
        !*/

        void async_copy_to_device(
        ); 
        /*!
            ensures
                - if (!device_ready()) then
                    - Begins asynchronously copying host data to the device once it is safe
                      to do so.  I.e. This function will wait until any previously
                      scheduled CUDA kernels, which are using the device() memory block,
                      have completed before transferring the new data to the device.
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

        bool host_ready (
        ) const;
        /*!
            ensures
                - returns true if and only if the host's copy of the data is current.  The
                  host's data is current if there aren't any modifications to the data
                  which were made on the device side that have yet to be copied to the
                  host.
        !*/

        bool device_ready (
        ) const; 
        /*!
            ensures
                - returns true if and only if the device's copy of the data is current.
                  The device's data is current if there aren't any modifications to the
                  data which were made on the host side that have yet to be copied to the
                  device.
        !*/

        const float* host(
        ) const;
        /*!
            ensures
                - returns a pointer to the host memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (!host_ready()) then
                    - copies the data from the device to the host, while this is happening
                      the call to host() blocks. 
                - #host_ready() == true 
        !*/

        float* host(
        );
        /*!
            ensures
                - returns a pointer to the host memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (!host_ready()) then
                    - copies the data from the device to the host, while this is happening
                      the call to host() blocks. 
                - #host_ready() == true 
                - #device_ready() == false
                  I.e. Marks the device side data as out of date so that the next call to
                  device() will perform a host to device transfer.  If you want to begin
                  the transfer immediately then you can call async_copy_to_device() after
                  calling host().
        !*/

        float* host_write_only(
        );
        /*!
            ensures
                - This function returns the same pointer as host(), except that it never
                  performs a device to host memory copy.  Instead, it immediately marks the
                  device side data as out of date, effectively discarding it.  Therefore,
                  the values in the data pointed to by host_write_only() are undefined and
                  you should only call host_write_only() if you are going to assign to
                  every memory location in the returned memory block.  
                - #host_ready() == true
                - #device_ready() == false 
        !*/

        const float* device(
        ) const;
        /*!
            requires
                - DLIB_USE_CUDA is #defined
            ensures
                - returns a pointer to the device memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (!device_ready()) then
                    - copies the data from the host to the device, while this is happening
                      the call to device() blocks. 
                - #device_ready() == true
        !*/

        float* device(
        );
        /*!
            requires
                - DLIB_USE_CUDA is #defined
            ensures
                - returns a pointer to the device memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (!device_ready()) then
                    - copies the data from the host to the device, while this is happening
                      the call to device() blocks. 
                - #host_ready() == false
                - #device_ready() == true
        !*/

        float* device_write_only(
        );
        /*!
            requires
                - DLIB_USE_CUDA is #defined
            ensures
                - This function returns the same pointer as device(), except that it never
                  performs a host to device memory copy.  Instead, it immediately marks the
                  host side data as out of date, effectively discarding it.  Therefore, the
                  values in the data pointed to by device_write_only() are undefined and
                  you should only call device_write_only() if you are going to assign to
                  every memory location in the returned memory block.  
                - #host_ready() == false 
                - #device_ready() == true 
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

    void memcpy (
        gpu_data& dest, 
        const gpu_data& src
    );
    /*!
        requires
            - dest.size() == src.size()
        ensures
            - Copies the data in src to dest.  If the device data is current (i.e.
              device_ready()==true) on both src and dest then the copy will happen entirely
              on the device side.
            - It doesn't matter what GPU device is selected by cudaSetDevice().  You can
              always copy gpu_data objects to and from each other regardless.
            - This function blocks until the copy has completed.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GPU_DaTA_ABSTRACT_H_

