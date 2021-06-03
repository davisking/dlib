// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TIMER_KERNEl_1_
#define DLIB_TIMER_KERNEl_1_

#include "../threads.h"
#include "../algs.h"
#include "../misc_api.h"
#include "timer_abstract.h"

namespace dlib
{

    template <
        typename T
        >
    class timer_heavy
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the timer_abstract.h interface.  It is very
                simple and uses only one thread which is always alive in a timer_heavy.
                The reason this object exists is for historical reasons.  Originally, the
                dlib::timer was a multi-implementation component and the timer_heavy was
                its first implementation.  It was superseded later by the more efficient
                dlib::timer.  However, timer_heavy is still around so that
                dlib::timer::kernel_1a has something to refer to.  This way, old client
                code which somehow depends on the same thread always calling a timer action
                function isn't going to be disrupted.


            INITIAL VALUE
                - running   == false
                - delay     == 1000
                - ao        == a pointer to the action_object()
                - af        == a pointer to the action_function()
                - m         == a mutex that locks everything in this class
                - s         == a signaler for mutex m
                - stop_running == false

            CONVENTION
                - running && !stop_running == is_running()
                - delay == delay_time()
                - *ao == action_object()
                - af == action_function()    

                - if (running) then
                    - there is a thread running
                - if (is_running()) then
                    - next_time_to_run == the time when the next execution of the action
                      function should occur.  (the time is given by ts.get_timestamp())

                - stop_running is used to tell the thread to quit.  If it is
                  set to true then the thread should end.
        !*/

    public:

        typedef void (T::*af_type)();

        timer_heavy(  
            T& ao_,
            af_type af_
        );

        virtual ~timer_heavy(
        );

        void clear(
        );

        af_type action_function (
        ) const;

        const T& action_object (
        ) const;

        T& action_object (
        );

        bool is_running (
        ) const;

        unsigned long delay_time (
        ) const;

        void set_delay_time (
            unsigned long milliseconds
        );
        
        void start (            
        );

        void stop (
        );

        void stop_and_wait (
        );

    private:

        void thread (
        );
        /*!
            requires
                - is run in its own thread
            ensures
                - calls the action function for the given timer object in the manner
                  specified by timer_kernel_abstract.h
        !*/

        // data members
        T& ao;
        const af_type af;
        unsigned long delay;
        mutex m;
        signaler s;

        bool running;
        bool stop_running;
        timestamper ts;
        uint64 next_time_to_run;

        // restricted functions
        timer_heavy(const timer_heavy<T>&);        // copy constructor
        timer_heavy<T>& operator=(const timer_heavy<T>&);    // assignment operator

    };    

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    // member function definitions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    
    template <
        typename T
        >
    timer_heavy<T>::
    timer_heavy(  
        T& ao_,
        af_type af_
    ) : 
        ao(ao_),
        af(af_),
        delay(1000),
        s(m),
        running(false),
        stop_running(false)
    {
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    timer_heavy<T>::
    ~timer_heavy(
    )
    {
        stop_and_wait();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void timer_heavy<T>::
    clear(
    )
    {
        m.lock();
        stop_running = true;
        delay = 1000;        
        s.broadcast();
        m.unlock();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    typename timer_heavy<T>::af_type timer_heavy<T>::
    action_function (
    ) const
    {
        return af;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    const T& timer_heavy<T>::
    action_object (
    ) const
    {
        return ao;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    T& timer_heavy<T>::
    action_object (
    )
    {
        return ao;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    bool timer_heavy<T>::
    is_running (
    ) const
    {
        auto_mutex M(m);
        return running && !stop_running;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    unsigned long timer_heavy<T>::
    delay_time (
    ) const
    {
        auto_mutex M(m);
        return delay;        
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void timer_heavy<T>::
    set_delay_time (
        unsigned long milliseconds
    )
    {
        m.lock();

        // if (is_running()) then we should adjust next_time_to_run
        if (running && !stop_running)
        {
            next_time_to_run -= delay*1000;
            next_time_to_run += milliseconds*1000;
        }

        delay = milliseconds;
        s.broadcast();
        m.unlock();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void timer_heavy<T>::
    start (            
    )
    {
        auto_mutex M(m);

        // if (is_running() == false) then reset the countdown to the next call 
        // to the action_function()
        if ( (running && !stop_running) == false)
            next_time_to_run = ts.get_timestamp() + delay*1000;

        stop_running = false;
        if (running == false)
        {
            running = true;

            // start the thread
            if (create_new_thread<timer_heavy,&timer_heavy::thread>(*this) == false)
            {
                running = false;
                throw dlib::thread_error("error creating new thread in timer_heavy::start");
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void timer_heavy<T>::
    stop (
    )
    {
        m.lock();
        stop_running = true;
        s.broadcast();
        m.unlock();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void timer_heavy<T>::
    thread (
    )
    {
        auto_mutex M(m);
        unsigned long delay_remaining;
        uint64 current_time = ts.get_timestamp();

        if (current_time < next_time_to_run)
            delay_remaining = static_cast<unsigned long>((next_time_to_run-current_time)/1000);
        else
            delay_remaining = 0;

        while (stop_running == false)
        {
            if (delay_remaining > 0)
                s.wait_or_timeout(delay_remaining);

            if (stop_running)
                break;            

            current_time = ts.get_timestamp();
            if (current_time < next_time_to_run)
            {
                // then we woke up too early so we should keep waiting
                delay_remaining = static_cast<unsigned long>((next_time_to_run-current_time)/1000);

                // rounding might make this be zero anyway.  So if it is
                // then we will say we have hit the next time to run.
                if (delay_remaining > 0)
                    continue;
            }

            // call the action function 
            m.unlock();
            (ao.*af)(); 
            m.lock();

            current_time = ts.get_timestamp();
            next_time_to_run = current_time + delay*1000;
            delay_remaining = delay;
        }
        running = false;
        stop_running = false;
        s.broadcast();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void timer_heavy<T>::
    stop_and_wait (
    )
    {
        m.lock();
        if (running)
        {
            // make the running thread terminate
            stop_running = true;

            s.broadcast();
            // wait for the thread to quit
            while (running)
                s.wait();          
        }
        m.unlock();
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_TIMER_KERNEl_1_

