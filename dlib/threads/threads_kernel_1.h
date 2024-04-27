// Copyright (C) 2003  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_THREADS_KERNEl_1_
#define DLIB_THREADS_KERNEl_1_

#ifdef DLIB_ISO_CPP_ONLY
#error "DLIB_ISO_CPP_ONLY is defined so you can't use this OS dependent code.  Turn DLIB_ISO_CPP_ONLY off if you want to use it."
#endif

#include "threads_kernel_abstract.h"

#include "../algs.h"
#include <condition_variable>
#include <mutex>
#include <chrono>


namespace dlib
{

// ----------------------------------------------------------------------------------------
    
    typedef unsigned long thread_id_type;

    thread_id_type get_thread_id();

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // mutex object
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    // forward declaration of signaler
    class signaler;

    class mutex
    {
    public:

        mutex (
        ) 
        {
        }

        ~mutex (
        ) {  }

        void lock (
        ) const { cs.lock(); }

        void unlock (
        ) const { cs.unlock(); }

    private:

        friend class signaler;

        mutable std::mutex cs;

        // restricted functions
        mutex(mutex&);        // copy constructor
        mutex& operator=(mutex&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // signaler object
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class signaler
    {

    public:
        signaler (
            const mutex& associated_mutex
        ) :
            m(associated_mutex)
        {           

        }

        ~signaler (
        ) { }

        void wait (
        ) const
        { 
            std::unique_lock<std::mutex> cs(m.cs, std::adopt_lock);
            cv.wait(cs);
            // Make sure we don't actually modify the mutex. Since the calling code will have locked
            // it and it should remain locked.
            cs.release();			
        }

        bool wait_or_timeout (
            unsigned long milliseconds
        ) const
        { 
            std::unique_lock<std::mutex> cs(m.cs, std::adopt_lock);
            auto status = cv.wait_until(cs, std::chrono::system_clock::now() + std::chrono::milliseconds(milliseconds));
            // Make sure we don't actually modify the mutex. Since the calling code will have locked
            // it and it should remain locked.
            cs.release();			
            return status == std::cv_status::no_timeout;
        }

        void signal (
        ) const 
        { 
            cv.notify_one();
        }

        void broadcast (
        ) const 
        { 
            cv.notify_all();
        }

        const mutex& get_mutex (
        ) const { return m; }

    private:

        mutable std::condition_variable cv;

        const mutex& m;

        // restricted functions
        signaler(signaler&);        // copy constructor
        signaler& operator=(signaler&);    // assignment operator
    };

// ----------------------------------------------------------------------------------------

    namespace threads_kernel_shared_helpers
    {
        bool spawn_thread (
            void (*funct)(void*),
            void* param
        );
        /*!
            is identical to create_new_thread() but just doesn't use any thread pooling.
        !*/
    }

// ----------------------------------------------------------------------------------------

}

#include "threads_kernel_shared.h"

#ifdef NO_MAKEFILE
#include "threads_kernel_1.cpp"
#endif

#endif // DLIB_THREADS_KERNEl_1_

