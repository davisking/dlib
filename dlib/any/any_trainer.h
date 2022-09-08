// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AnY_TRAINER_H_
#define DLIB_AnY_TRAINER_H_

#include "any.h"
#include "any_decision_function.h"
#include "any_trainer_abstract.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename sample_type_,
        typename scalar_type_ = double
        >
    class any_trainer : public any
    {
    public:
        typedef sample_type_ sample_type;
        typedef scalar_type_ scalar_type;
        typedef default_memory_manager mem_manager_type;
        typedef any_decision_function<sample_type, scalar_type> trained_function_type;

        using any::any;

        template <
            class T,
            class T_ = std::decay_t<T>,
            std::enable_if_t<!std::is_same<T_,any_trainer>::value, bool> = true
        >
        any_trainer (
            T&& item
        ) : any{std::forward<T>(item)},
            train_func{[](
                const void* ptr,
                const std::vector<sample_type>& samples,
                const std::vector<scalar_type>& labels
            ) -> trained_function_type {
                const T_& f = *reinterpret_cast<const T_*>(ptr);
                return f.train(samples, labels);
            }}
        {
        }

        template <
            class T,
            class T_ = std::decay_t<T>,
            std::enable_if_t<!std::is_same<T_,any_trainer>::value, bool> = true
        >
        any_trainer& operator= (
            T&& item
        )
        {
            if (contains<T_>())
                unsafe_get<T_>() = std::forward<T>(item);
            else
                *this = std::move(any_trainer{std::forward<T>(item)});
            return *this;
        }

        trained_function_type train (
            const std::vector<sample_type>& samples,
            const std::vector<scalar_type>& labels
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_empty() == false,
                "\t trained_function_type any_trainer::train()"
                << "\n\t You can't call train() on an empty any_trainer"
                << "\n\t this: " << this
                );

            return train_func(ptr, samples, labels);
        }

        void swap (
            any_trainer& item
        )
        {
            any_trainer tmp{std::move(*this)};
            *this = std::move(item);
            item  = std::move(tmp);
        }

    private:

        trained_function_type (*train_func) (
            const void* self,
            const std::vector<sample_type>& samples,
            const std::vector<scalar_type>& labels
        ) = nullptr;
    };

// ----------------------------------------------------------------------------------------
}


#endif // DLIB_AnY_TRAINER_H_




