// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AnY_DECISION_FUNCTION_Hh_
#define DLIB_AnY_DECISION_FUNCTION_Hh_

#include "any_decision_function_abstract.h"
#include "any.h"
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename sample_type_,
        typename result_type_ = double
        >
    class any_decision_function
    {

    public:
        using sample_type = sample_type_;
        using result_type = result_type_;
        using mem_manager_type = default_memory_manager;

        any_decision_function()                                                 = default;
        any_decision_function(const any_decision_function& other)               = default;
        any_decision_function& operator=(const any_decision_function& other)    = default;
        any_decision_function(any_decision_function&& other)                    = default;
        any_decision_function& operator=(any_decision_function&& other)         = default;

        template <
            class T,
            class T_ = std::decay_t<T>,
            std::enable_if_t<!std::is_same<T_,any_decision_function>::value, bool> = true
        >
        any_decision_function (
            T&& item
        ) : storage{std::forward<T>(item)},
            evaluate_func{[](const void* ptr, const sample_type& samp) -> result_type {
                const T_& f = *reinterpret_cast<const T_*>(ptr);
                return f(samp);
            }}
        {
        }

        template <
            class T,
            class T_ = std::decay_t<T>,
            std::enable_if_t<!std::is_same<T_,any_decision_function>::value, bool> = true
        >
        any_decision_function& operator= (
            T&& item
        )
        {
            if (contains<T_>())
                storage.unsafe_get<T_>() = std::forward<T>(item);
            else
                *this = std::move(any_decision_function{std::forward<T>(item)});
            return *this;
        }

        result_type operator() (
            const sample_type& item
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_empty() == false,
                "\t result_type any_decision_function::operator()"
                << "\n\t You can't call operator() on an empty any_decision_function"
                << "\n\t this: " << this
                );

            return evaluate_func(storage.ptr, item);
        }

        template<typename T>
        bool contains() const { return storage.contains<T>();}
        bool is_empty() const { return storage.is_empty(); }
        void clear()          { storage.clear(); }
        void swap (any_decision_function& item) { std::swap(storage, item.storage); }

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
                *this = T{};
            return storage.unsafe_get<T>();
        }

    private:
        te::storage_heap storage;
        result_type (*evaluate_func)(const void*, const sample_type&);
    };

// ----------------------------------------------------------------------------------------

}


#endif // DLIB_AnY_DECISION_FUNCTION_Hh_


