// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AnY_H_
#define DLIB_AnY_H_

#include "any_abstract.h"
#include <memory>
#include "storage.h"

namespace dlib
{
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
            std::enable_if_t<!std::is_same<std::decay_t<T>, any>::value, bool> = true
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
            if (contains<T_>())
                storage.unsafe_get<T_>() = std::forward<T>(item);
            else
                *this = std::move(any{std::forward<T>(item)});
            return *this;
        }

        bool is_empty() const { return storage.is_empty(); }
        void clear()          { storage.clear(); }
        void swap (any& item) { std::swap(*this, item); }

        template<typename T>      bool contains() const { return storage.contains<T>();}
        template <typename T>       T& cast_to()        { return storage.cast_to<T>(); }
        template <typename T> const T& cast_to() const  { return storage.cast_to<T>(); }
        template <typename T>       T& get()            { return storage.get<T>(); }

    private:
        te::storage_heap storage;
    };

// ----------------------------------------------------------------------------------------

    template <typename T> T& any_cast(any& a) { return a.cast_to<T>(); }
    template <typename T> const T& any_cast(const any& a) { return a.cast_to<T>(); }

// ----------------------------------------------------------------------------------------

}


#endif // DLIB_AnY_H_



