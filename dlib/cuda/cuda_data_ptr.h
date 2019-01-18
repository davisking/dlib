// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDA_DATA_PTR_H_
#define DLIB_DNN_CuDA_DATA_PTR_H_

#include "../assert.h"

#ifdef DLIB_USE_CUDA

#include <memory>
#include <vector>

namespace dlib
{
    namespace cuda
    {

    // ------------------------------------------------------------------------------------

        class cuda_data_void_ptr
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a block of memory on a CUDA device.  
            !*/
        public:

            cuda_data_void_ptr() = default;

            cuda_data_void_ptr(size_t n); 
            /*!
                ensures
                    - This object will allocate a device memory buffer of n bytes.
                    - #size() == n
            !*/

            void* data() { return pdata.get(); }
            const void* data() const { return pdata.get(); }
            operator void*() { return pdata.get(); }
            operator const void*() const { return pdata.get(); }

            void reset() { pdata.reset(); }

            size_t size() const { return num; }
            /*!
                ensures
                    - returns the length of this buffer, in bytes.
            !*/

            cuda_data_void_ptr operator+ (size_t offset) const 
            /*!
                requires
                    - offset < size()
                ensures
                    - returns a pointer that is offset by the given amount.
            !*/
            { 
                DLIB_CASSERT(offset < num);
                cuda_data_void_ptr temp;
                temp.num = num-offset;
                temp.pdata = std::shared_ptr<void>(pdata, ((char*)pdata.get())+offset);
                return temp;
            }

        private:

            size_t num = 0;
            std::shared_ptr<void> pdata;
        };

        inline cuda_data_void_ptr operator+(size_t offset, const cuda_data_void_ptr& rhs) { return rhs+offset; }

    // ------------------------------------------------------------------------------------

        void memcpy(
            void* dest,
            const cuda_data_void_ptr& src
        );
        /*!
            requires
                - dest == a pointer to at least src.size() bytes on the host machine.
            ensures
                - copies the GPU data from src into dest.
                - This routine is equivalent to performing: memcpy(dest,src,src.size())
        !*/

        void memcpy(
            void* dest,
            const cuda_data_void_ptr& src,
            const size_t num
        );
        /*!
            requires
                - dest == a pointer to at least num bytes on the host machine.
                - num <= src.size()
            ensures
                - copies the GPU data from src into dest.  Copies only the first num bytes
                  of src to dest.
        !*/

    // ------------------------------------------------------------------------------------

        void memcpy(
            cuda_data_void_ptr dest, 
            const void* src
        );
        /*!
            requires
                - dest == a pointer to at least src.size() bytes on the host machine.
            ensures
                - copies the host data from src to the GPU memory buffer dest.
                - This routine is equivalent to performing: memcpy(dest,src,dest.size())
        !*/

        void memcpy(
            cuda_data_void_ptr dest, 
            const void* src,
            const size_t num
        );
        /*!
            requires
                - dest == a pointer to at least num bytes on the host machine.
                - num <= dest.size()
            ensures
                - copies the host data from src to the GPU memory buffer dest.  Copies only
                  the first num bytes of src to dest.
        !*/

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        template <typename T>
        class cuda_data_ptr
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a block of memory on a CUDA device.   It is just a type safe
                    version of cuda_data_void_ptr.
            !*/

        public:

            static_assert(std::is_standard_layout<T>::value, "You can only create basic standard layout types on the GPU");

            cuda_data_ptr() = default;
            cuda_data_ptr(size_t n) : num(n)
            /*!
                ensures
                    - This object will allocate a device memory buffer of n T objects.
                    - #size() == n
            !*/
            {
                if (n == 0)
                    return;

                pdata = cuda_data_void_ptr(n*sizeof(T));
            }

            T* data() { return (T*)pdata.data(); }
            const T* data() const { return (T*)pdata.data(); }

            operator T*() { return (T*)pdata.data(); }
            operator const T*() const { return (T*)pdata.data(); }

            void reset() { pdata.reset(); }

            size_t size() const { return num; }


            friend void memcpy(
                std::vector<T>& dest,
                const cuda_data_ptr& src
            )
            {
                dest.resize(src.size());
                if (src.size() != 0)
                    memcpy(dest.data(), src.pdata);
            }

            friend void memcpy(
                cuda_data_ptr& dest,
                const std::vector<T>& src
            )
            {
                if (src.size() != dest.size())
                    dest = cuda_data_ptr<T>(src.size());

                if (dest.size() != 0)
                    memcpy(dest.pdata, src.data());
            }

            friend void memcpy(
                cuda_data_ptr& dest,
                const float* src
            )
            {
                memcpy(dest.pdata, src);
            }

            friend void memcpy(
                float* dest, 
                const cuda_data_ptr& src
            )
            {
                memcpy(dest, src.pdata);
            }

        private:

            size_t num = 0;
            cuda_data_void_ptr pdata;
        };

    // ------------------------------------------------------------------------------------

        class resizable_cuda_buffer
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a block of memory on a CUDA device that will be automatically
                    resized if requested size is larger than allocated.
            !*/
        public:
            cuda_data_void_ptr get(size_t size)
            /*!
                ensures
                    - This object will return the buffer of requested size or larger.
                    - buffer.size() >= size
                    - Client code should not hold the returned cuda_data_void_ptr for long
                      durations, but instead should call get() whenever the buffer is
                      needed.  Doing so ensures that multiple buffers are not kept around
                      in the event of a resize.
            !*/
            {
                if (buffer.size() < size)
                {
                    buffer.reset();
                    buffer = cuda_data_void_ptr(size);
                }
                return buffer;
            }
        private:
            cuda_data_void_ptr buffer;
        };

    // ----------------------------------------------------------------------------------------

        std::shared_ptr<resizable_cuda_buffer> device_global_buffer(
        );
        /*!
            ensures
                - Returns a pointer to a globally shared CUDA memory buffer on the
                  currently selected CUDA device.  The buffer is also thread local.  So
                  each host thread will get its own buffer.  You can use this global buffer
                  as scratch space for CUDA computations that all take place on the default
                  stream.  Using it in this way ensures that there aren't any race conditions
                  involving the use of the buffer.
                - The global buffer is deallocated once all references to it are
                  destructed.  It will be reallocated as required.  So if you want to avoid
                  these reallocations then hold a copy of the shared_ptr returned by this
                  function.
        !*/

    // ----------------------------------------------------------------------------------------

    }
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDA_DATA_PTR_H_

