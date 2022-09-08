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

    class any
    {
    public:
        any()                               = default;
        any(const any& other)               = default;
        any& operator=(const any& other)    = default;
        any(any&& other)                    = default;
        any& operator=(any&& other)         = default;

        template<
            typename T,
            typename T_ = std::decay_t<T>,
            std::enable_if_t<!std::is_same<T_, any>::value, bool> = true
        >
        any(T&& item)
        : storage{std::forward<T>(item)}
        {
        }

        template<
            typename T,
            typename T_ = std::decay_t<T>,
            std::enable_if_t<!std::is_same<T_, any>::value, bool> = true
        >
        any& operator=(T&& item)
        {
            get<T_>() = std::forward<T>(item);
            return *this;
        }

        template<typename T>
        bool contains() const { return storage.contains<T>();}
        bool is_empty() const { return storage.is_empty(); }
        void clear()          { storage.clear(); }
        void swap (any& item) { std::swap(storage, item.storage); }

        template <typename T>
        T& cast_to(
        ) 
        {
            if (!storage.contains<T>())
                throw bad_any_cast{};
            return storage.unsafe_get<T>();
        }

        template <typename T>
        const T& cast_to(
        ) const
        {
            if (!storage.contains<T>())
                throw bad_any_cast{};
            return storage.unsafe_get<T>();
        }

        template <typename T>
        T& get(
        ) 
        {
            if (!storage.contains<T>())
                storage = T{};
            return storage.unsafe_get<T>();
        }

    private:
        te::storage_heap storage;
    };

// ----------------------------------------------------------------------------------------

    template <typename T> T& any_cast(any& a) { return a.cast_to<T>(); }
    template <typename T> const T& any_cast(const any& a) { return a.cast_to<T>(); }

// ----------------------------------------------------------------------------------------

}


#endif // DLIB_AnY_H_



