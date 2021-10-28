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
        using return_type   = typename impl::result_of<Function(Args...)>::type;
        using task_type     = std::packaged_task<return_type()>;
        auto task = std::make_shared<task_type>(std::bind(std::forward<Function>(f), std::forward<Args>(args)...));
        auto ret  = task->get_future();
        tp.add_task_by_value([task]() {(*task)();});
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



