// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AsYNC_Hh_
#define DLIB_AsYNC_Hh_ 

// C++11 things don't work in old versions of visual studio 
#if !defined( _MSC_VER) ||  _MSC_VER >= 1900

#include "async_abstract.h"
#include "thread_pool_extension.h"
#include <future>
#include <functional>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename T> struct selector {};

        template <typename T, typename U, typename V>
        void call_prom_set_value(
            T& prom,
            U& fun,
            selector<V> 
        )
        {
            prom.set_value(fun());
        }

        template <typename T, typename U>
        void call_prom_set_value(
            T& prom,
            U& fun,
            selector<void>
        )
        {
            fun();
            prom.set_value();
        }

        template <typename> struct result_of;

#if (__cplusplus >= 201703L ||                          \
     (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)) && \
    __cpp_lib_is_invocable >= 201703L
        template <typename F, typename... Args>
        struct result_of<F(Args...)> : std::invoke_result<F, Args...> {};
#else
        template <typename F, typename... Args>
        struct result_of<F(Args...)>
                : std::result_of<F&&(Args&&...)> {};
#endif
    }

// ----------------------------------------------------------------------------------------

    thread_pool& default_thread_pool();

// ----------------------------------------------------------------------------------------

    template < 
        typename Function, 
        typename ...Args
        >
    std::future<typename impl::result_of<Function(Args...)>::type> async(
        thread_pool& tp, 
        Function&& f, 
        Args&&... args 
    )
    {
        auto prom = std::make_shared<std::promise<typename impl::result_of<Function(Args...)>::type>>();
        std::future<typename impl::result_of<Function(Args...)>::type> ret = prom->get_future();
        using bind_t = decltype(std::bind(std::forward<Function>(f), std::forward<Args>(args)...));
        auto fun = std::make_shared<bind_t>(std::bind(std::forward<Function>(f), std::forward<Args>(args)...));
        tp.add_task_by_value([fun, prom]()
        { 
            try
            {
                impl::call_prom_set_value(*prom, *fun, impl::selector<typename impl::result_of<Function(Args...)>::type>());
            }
            catch(...)
            {
                prom->set_exception(std::current_exception());
            }
        });
        return ret;
    }

// ----------------------------------------------------------------------------------------

    template < 
        typename Function, 
        typename ...Args
        >
    std::future<typename impl::result_of<Function(Args...)>::type> async(
        Function&& f, 
        Args&&... args 
    )
    {
        return async(default_thread_pool(), std::forward<Function>(f), std::forward<Args>(args)...);
    }

}

// ----------------------------------------------------------------------------------------

#ifdef NO_MAKEFILE
#include "async.cpp"
#endif

#endif
#endif // DLIB_AsYNC_Hh_



