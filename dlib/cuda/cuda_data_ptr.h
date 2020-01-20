// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDA_DATA_PTR_H_
#define DLIB_DNN_CuDA_DATA_PTR_H_

#include "../assert.h"

#ifdef DLIB_USE_CUDA

#include <memory>
#include <vector>
#include <type_traits>

namespace dlib
{
    namespace cuda
    {

    // ------------------------------------------------------------------------------------

        class cuda_data_void_ptr;
        class weak_cuda_data_void_ptr 
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is just like a std::weak_ptr version of cuda_data_void_ptr.  It allows you
                    to hold a non-owning reference to a cuda_data_void_ptr.
            !*/
        public:
            weak_cuda_data_void_ptr() = default;

            weak_cuda_data_void_ptr(const cuda_data_void_ptr& ptr);

            void reset() { pdata.reset(); num = 0; }

            cuda_data_void_ptr lock() const;
            /*!
                ensures
                    - if (the memory block referenced by this object hasn't been deleted) then
                        - returns a cuda_data_void_ptr referencing that memory block
                    - else
                        - returns a default initialized cuda_data_void_ptr (i.e. an empty one).
            !*/

        private:
            size_t num = 0;
            std::weak_ptr<void> pdata;
        };

    // ----------------------------------------------------------------------------------------

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

            void shrink(size_t new_size) 
            /*!
                requires
                    - new_size <= num
                ensures
                    - #size() == new_size
                    - Doesn't actually deallocate anything, just changes the size() metadata to a
                      smaller number and only for this instance of the pointer.
            !*/
            {
                DLIB_CASSERT(new_size <= num);
                num = new_size;
            }

        private:

            friend class weak_cuda_data_void_ptr;
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

            cuda_data_ptr(
                const cuda_data_ptr<typename std::remove_const<T>::type> &other
            ) : num(other.num), pdata(other.pdata) {}
            /*!
                ensures
                    - *this is a copy of other.  This version of the copy constructor allows
                      assigning non-const pointers to const ones.  For instance, converting from
                      cuda_data_ptr<float> to cuda_data_ptr<const float>.
            !*/

            T* data() { return (T*)pdata.data(); }
            const T* data() const { return (T*)pdata.data(); }

            operator T*() { return (T*)pdata.data(); }
            operator const T*() const { return (T*)pdata.data(); }

            void reset() { pdata.reset(); }

            size_t size() const { return num; }
            /*!
                ensures
                    - returns the number of T instances pointed to by *this.
            !*/

            operator cuda_data_void_ptr() const 
            /*!
                ensures
                    - returns *this as a cuda_data_void_ptr.  Importantly, the returned size() will
                      reflect the number of bytes referenced by *this.  To be clear, let P be the
                      returned pointer.  Then:
                        - P.get() == get()
                        - P.size() == size() * sizeof(T)
            !*/
            { 
                cuda_data_void_ptr temp = pdata;
                temp.shrink(size() * sizeof(T));
                return temp;
            }

        private:
            template <typename U>
            friend cuda_data_ptr<U> static_pointer_cast(const cuda_data_void_ptr &ptr);
            template <typename U>
            friend cuda_data_ptr<U> static_pointer_cast(const cuda_data_void_ptr &ptr, size_t num);
            template <typename U>
            friend class cuda_data_ptr;

            size_t num = 0;
            cuda_data_void_ptr pdata;
        };

        template <typename T>
        cuda_data_ptr<T> static_pointer_cast(const cuda_data_void_ptr &ptr) 
        {
            DLIB_CASSERT(ptr.size() % sizeof(T) == 0, 
                "Size of memory buffer in ptr doesn't match sizeof(T). "
                << "\nptr.size(): "<< ptr.size() 
                << "\nsizeof(T): "<< sizeof(T));
            cuda_data_ptr<T> result;
            result.pdata = ptr;
            result.num = ptr.size() / sizeof(T);
            return result;
        }

        template <typename T>
        cuda_data_ptr<T> static_pointer_cast(const cuda_data_void_ptr &ptr, size_t num) 
        {
            DLIB_CASSERT(num*sizeof(T) <= ptr.size(), 
                "Size of memory buffer in ptr isn't big enough to represent this many T objects. "
                << "\nnum: "<< num 
                << "\nnum*sizeof(T): "<< num*sizeof(T)
                << "\nsizeof(T): "<< sizeof(T)
                << "\nptr.size(): "<< ptr.size());

            cuda_data_ptr<T> result;
            result.pdata = ptr;
            result.num = num;
            return result;
        }

        template <typename T>
        void memcpy(std::vector<T>& dest, const cuda_data_ptr<T>& src)
        {
            dest.resize(src.size());
            if (src.size() != 0)
                memcpy(dest.data(), static_cast<cuda_data_void_ptr>(src));
        }

        template <typename T>
        void memcpy(cuda_data_ptr<T>& dest, const std::vector<T>& src)
        {
            if (src.size() != dest.size())
                dest = cuda_data_ptr<T>(src.size());

            if (dest.size() != 0)
                memcpy(static_cast<cuda_data_void_ptr>(dest), src.data());
        }

        template <typename T>
        void memcpy(cuda_data_ptr<T>& dest, const T* src)
        {
            memcpy(static_cast<cuda_data_void_ptr>(dest), src);
        }
        template <typename T>
        void memcpy(cuda_data_ptr<T>& dest, const T* src, size_t num)
        {
            DLIB_CASSERT(num <= dest.size());
            memcpy(static_cast<cuda_data_void_ptr>(dest), src, num*sizeof(T));
        }

        template <typename T>
        void memcpy(T* dest, const cuda_data_ptr<T>& src)
        {
            memcpy(dest, static_cast<cuda_data_void_ptr>(src));
        }
        template <typename T>
        void memcpy(T* dest, const cuda_data_ptr<T>& src, size_t num)
        {
            DLIB_CASSERT(num <= src.size());
            memcpy(dest, static_cast<cuda_data_void_ptr>(src), num*sizeof(T));
        }

    // ------------------------------------------------------------------------------------

        cuda_data_void_ptr device_global_buffer(size_t size);
        /*!
            ensures
                - Returns a pointer to a globally shared CUDA memory buffer on the
                  currently selected CUDA device.  The buffer is also thread local.  So
                  each host thread will get its own buffer.  You can use this global buffer
                  as scratch space for CUDA computations that all take place on the default
                  stream.  Using it in this way ensures that there aren't any race conditions
                  involving the use of the buffer.
                - The returned pointer will point to at least size bytes.  It may point to more.
                - The global buffer is deallocated once all references to it are destructed.
                  However, if device_global_buffer() is called before then with a size <= the last
                  size requested, then the previously returned global buffer pointer is returned.
                  This avoids triggering expensive CUDA reallocations.  So if you want to avoid
                  these reallocations then hold a copy of the pointer returned by this function.
                  However, as a general rule, client code should not hold the returned
                  cuda_data_void_ptr for long durations, but instead should call
                  device_global_buffer() whenever the buffer is needed, and overwrite the previously
                  returned pointer with the new pointer.  Doing so ensures multiple buffers are not
                  kept around in the event that multiple sized buffers are requested.  To explain
                  this, consider this code, assumed to execute at program startup:
                    auto ptr1 = device_global_buffer(1);
                    auto ptr2 = device_global_buffer(2);
                    auto ptr3 = device_global_buffer(3);
                  since the sizes increased at each call 3 separate buffers were allocated.  First
                  one of size 1, then of size 2, then of size 3.  If we then executed:
                    ptr1 = device_global_buffer(1);
                    ptr2 = device_global_buffer(2);
                    ptr3 = device_global_buffer(3);
                  all three of these pointers would now point to the same buffer, since the smaller
                  requests can be satisfied by returning the size 3 buffer in each case.
        !*/

    // ----------------------------------------------------------------------------------------

    }
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDA_DATA_PTR_H_

