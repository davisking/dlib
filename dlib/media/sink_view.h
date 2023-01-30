// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_SINK_VIEW
#define DLIB_SINK_VIEW

#include <utility>
#include <type_traits>
#include "../assert.h"

namespace dlib
{
    namespace ffmpeg
    {

// -----------------------------------------------------------------------------------------------------

        class sink_view
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a view type (think std::string_view, or std::span) that
                    wraps a buffer, output stream or file which can be used by dlib::ffmpeg::encoder
                    to push encoded data.
            !*/

        public:
            using push_func = void (*)(void* ptr, std::size_t ndata, const char* data);    

            sink_view() = default;
            /*!
                ensures
                    - #is_empty() == true
            !*/

            // This object has the same copy semantics as a void*.
            sink_view(const sink_view&)             = default;
            sink_view& operator=(const sink_view&)  = default;

            sink_view(sink_view&& other) noexcept
            /*!
                ensures
                    - #is_empty() == other.is_empty()
                    - #other.is_empty() == true
            !*/
            : ptr{std::exchange(other.ptr, nullptr)},
              func{std::exchange(other.func, nullptr)}
            {
            }

            sink_view& operator=(sink_view&& other) noexcept
            /*!
                ensures
                    - #is_empty() == other.is_empty()
                    - #other.is_empty() == true
            !*/
            {
                if (this != &other)
                {
                    ptr     = std::exchange(other.ptr, nullptr);
                    func    = std::exchange(other.func, nullptr);
                }
                return *this;
            }

            sink_view(
                void*       ptr_,
                push_func   func_
            ) 
            /*!
                requires
                    - ptr_ != nullptr
                    - func_ != nullptr
                ensures
                    - #is_empty() == false
            !*/
            : ptr{ptr_},
              func{func_}
            {
                DLIB_ASSERT(!is_empty(), "don't pass nullptr for either ptr_ or func_");
            }

            bool is_empty() const noexcept
            /*!
                ensures
                    - true if underlying ptr and erased function are not null
            !*/
            {
                return ptr == nullptr || func == nullptr;
            }

            void push(std::size_t ndata, const char* data)
            /*!
                requires
                    - is_empty() == false
                ensures
                    - adds all of data to the underlying buffer/stream/file/...
            !*/
            {
                DLIB_ASSERT(!is_open(), "this is an empty view");
                func(ptr, ndata, data);
            }

        private:
            void*       ptr{nullptr};
            push_func   func{nullptr};
        };

// -----------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_SINK_VIEW