// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AnY_H_
#define DLIB_AnY_H_

#include "any_abstract.h"
#include <memory>
#include "../te.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class bad_any_cast : public std::bad_cast 
    {
    public:
          virtual const char * what() const throw()
          {
              return "bad_any_cast";
          }
    };

// ----------------------------------------------------------------------------------------

    class any : public te::storage_heap
    {
    public:
        using te::storage_heap::storage_heap;

        template <typename T>
        T& cast_to(
        ) 
        {
            if (!contains<T>())
                throw bad_any_cast{};
            return unsafe_get<T>();
        }

        template <typename T>
        const T& cast_to(
        ) const
        {
            if (!contains<T>())
                throw bad_any_cast{};
            return unsafe_get<T>();
        }

        template <typename T>
        T& get(
        ) 
        {
            if (!contains<T>())
                *this = T{};
            return unsafe_get<T>();
        }

        void swap (
            any& item
        )
        {
            any tmp{std::move(*this)};
            *this = std::move(item);
            item  = std::move(tmp);
        }

    private:
    };

// ----------------------------------------------------------------------------------------

    template <typename T> T& any_cast(any& a) { return a.cast_to<T>(); }
    template <typename T> const T& any_cast(const any& a) { return a.cast_to<T>(); }

// ----------------------------------------------------------------------------------------

}


#endif // DLIB_AnY_H_



