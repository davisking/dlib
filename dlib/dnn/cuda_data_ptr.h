// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDA_DATA_PTR_H_
#define DLIB_DNN_CuDA_DATA_PTR_H_

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

        private:

            size_t num = 0;
            std::shared_ptr<void> pdata;
        };

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
        !*/

    // ------------------------------------------------------------------------------------

        void memcpy(
            cuda_data_void_ptr& dest, 
            const void* src
        );
        /*!
            requires
                - dest == a pointer to at least src.size() bytes on the host machine.
            ensures
                - copies the host data from src to the GPU memory buffer dest.
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
                cuda_data_ptr& src,
                const std::vector<T>& dest
            )
            {
                if (dest.size() != src.size())
                    dest = cuda_data_ptr<T>(src.size());

                if (src.size() != 0)
                    memcpy(src.pdata, dest.data());
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
                    - This object will return the buffer of requested size of larger
                    - buffer.size() >= size
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

    }
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDA_DATA_PTR_H_

