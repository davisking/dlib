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
    class any_decision_function : public any
    {

    public:

        typedef sample_type_ sample_type;
        typedef result_type_ result_type;
        typedef default_memory_manager mem_manager_type;

        using any::any;

        template <
            class T,
            class T_ = std::decay_t<T>,
            std::enable_if_t<!std::is_same<T_,any_decision_function>::value, bool> = true
        >
        any_decision_function (
            T&& item
        ) : any{std::forward<T>(item)},
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
                unsafe_get<T_>() = std::forward<T>(item);
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

            return evaluate_func(ptr, item);
        }

    private:
        result_type (*evaluate_func)(const void*, const sample_type&);
    };

// ----------------------------------------------------------------------------------------

}


#endif // DLIB_AnY_DECISION_FUNCTION_Hh_


