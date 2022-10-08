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
    class any_trainer
    {
    public:
        using sample_type = sample_type_;
        using scalar_type = scalar_type_;
        using mem_manager_type = default_memory_manager;
        using trained_function_type = any_decision_function<sample_type, scalar_type>;

        any_trainer()                                       = default;
        any_trainer(const any_trainer& other)               = default;
        any_trainer& operator=(const any_trainer& other)    = default;
        any_trainer(any_trainer&& other)                    = default;
        any_trainer& operator=(any_trainer&& other)         = default;

        template <
            class T,
            class T_ = std::decay_t<T>,
            std::enable_if_t<!std::is_same<T_,any_trainer>::value, bool> = true
        >
        any_trainer (
            T&& item
        ) : storage{std::forward<T>(item)},
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
                storage.unsafe_get<T_>() = std::forward<T>(item);
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

            return train_func(storage.get_ptr(), samples, labels);
        }

        bool is_empty() const { return storage.is_empty(); }
        void clear()          { storage.clear(); }
        void swap (any_trainer& item) { std::swap(*this, item); }

        template <typename T>     bool contains() const { return storage.contains<T>();}
        template <typename T>       T& cast_to()        { return storage.cast_to<T>(); }
        template <typename T> const T& cast_to() const  { return storage.cast_to<T>(); }
        template <typename T>       T& get()            { return storage.get<T>(); }

    private:
        te::storage_heap storage;
        trained_function_type (*train_func) (
            const void* self,
            const std::vector<sample_type>& samples,
            const std::vector<scalar_type>& labels
        ) = nullptr;
    };

// ----------------------------------------------------------------------------------------

    template <typename T, typename U, typename V> 
    T& any_cast(any_trainer<U,V>& a) { return a.template cast_to<T>(); }

    template <typename T, typename U, typename V> 
    const T& any_cast(const any_trainer<U,V>& a) { return a.template cast_to<T>(); }

// ----------------------------------------------------------------------------------------

}


#endif // DLIB_AnY_TRAINER_H_




