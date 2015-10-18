// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_GPU_DaTA_H_
#define DLIB_GPU_DaTA_H_

#include <memory>
#include "cuda_errors.h"
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class gpu_data 
    {
        /*!
            CONVENTION
                - if (size() != 0) then
                    - data_host == a pointer to size() floats in CPU memory.
                - if (data_device) then 
                    - data_device == a pointer to size() floats in device memory.

                - if (there might be an active transfer between host and device) then
                    - have_active_transfer == true

                - We use the host_current and device_current bools to keep track of which
                  copy of the data (or both) are most current.  e.g. if the CPU has
                  modified the tensor and it hasn't been copied to the device yet then
                  host_current==true and device_current == false.


            THREAD SAFETY
                This object is not thread-safe.  Don't touch it from multiple threads as the
                same time.
        !*/
    public:

        gpu_data(
        ) : data_size(0), host_current(true), device_current(true),have_active_transfer(false)
        {
        }

        // Not copyable
        gpu_data(const gpu_data&) = delete;
        gpu_data& operator=(const gpu_data&) = delete;

        // but is movable
        gpu_data(gpu_data&&) = default;
        gpu_data& operator=(gpu_data&&) = default;


#ifdef DLIB_USE_CUDA
        void async_copy_to_device(); 
        void set_size(size_t new_size);
#else
        // Note that calls to host() or device() will block until any async transfers are complete.
        void async_copy_to_device(){}

        void set_size(size_t new_size)
        {
            if (new_size == 0)
            {
                data_size = 0;
                host_current = true;
                device_current = true;
                data_host.reset();
                data_device.reset();
            }
            else if (new_size != data_size)
            {
                data_size = new_size;
                host_current = true;
                device_current = true;
                data_host.reset(new float[new_size], std::default_delete<float[]>());
                data_device.reset();
            }
        }
#endif

        const float* host() const 
        { 
            copy_to_host();
            return data_host.get(); 
        }

        float* host() 
        {
            copy_to_host();
            device_current = false;
            return data_host.get(); 
        }

        const float* device() const 
        { 
#ifndef DLIB_USE_CUDA
            DLIB_CASSERT(false, "CUDA NOT ENABLED");
#endif
            copy_to_device();
            return data_device.get(); 
        }

        float* device() 
        {
#ifndef DLIB_USE_CUDA
            DLIB_CASSERT(false, "CUDA NOT ENABLED");
#endif
            copy_to_device();
            host_current = false;
            return data_device.get(); 
        }

        size_t size() const { return data_size; }


    private:

#ifdef DLIB_USE_CUDA
        void copy_to_device() const;
        void copy_to_host() const;
        void wait_for_transfer_to_finish() const;
#else
        void copy_to_device() const{}
        void copy_to_host() const{}
        void wait_for_transfer_to_finish() const{}
#endif


        size_t data_size;
        mutable bool host_current;
        mutable bool device_current;
        mutable bool have_active_transfer;

        std::shared_ptr<float> data_host;
        std::shared_ptr<float> data_device;
        std::shared_ptr<void> cuda_stream;
    };

    inline void serialize(const gpu_data& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.size(), out);
        auto data = item.host();
        for (size_t i = 0; i < item.size(); ++i)
            serialize(data[i], out);
    }

    inline void deserialize(gpu_data& item, std::istream& in)
    {
        int version;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::gpu_data.");
        size_t s;
        deserialize(s, in);
        item.set_size(s);
        auto data = item.host();
        for (size_t i = 0; i < item.size(); ++i)
            deserialize(data[i], in);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GPU_DaTA_H_

