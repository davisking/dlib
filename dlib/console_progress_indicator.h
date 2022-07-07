// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CONSOLE_PROGRESS_INDiCATOR_Hh_
#define DLIB_CONSOLE_PROGRESS_INDiCATOR_Hh_

#include <cmath>
#include <chrono>
#include <limits>
#include <iostream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class console_progress_indicator
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for reporting how long a task will take
                to complete.  

                For example, consider the following bit of code:

                    console_progress_indicator pbar(100)
                    for (int i = 1; i <= 100; ++i)
                    {
                        pbar.print_status(i);
                        long_running_operation();
                    }

                The above code will print a message to the console each iteration
                which shows the current progress and how much time is remaining until
                the loop terminates.
        !*/

    public:

        inline explicit console_progress_indicator (
            double target_value 
        ); 
        /*!
            ensures
                - #target() == target_value
        !*/

        inline void reset (
            double target_value
        );
        /*!
            ensures
                - #target() == target_value
                - performs the equivalent of:
                    *this = console_progress_indicator(target_value)
                    (i.e. resets this object with a new target value)

        !*/

        inline double target (
        ) const;
        /*!
            ensures
                - This object attempts to measure how much time is
                  left until we reach a certain targeted value.  This
                  function returns that targeted value.
        !*/

        inline bool print_status (
            double cur,
            bool always_print = false,
            std::ostream& out = std::clog
        );
        /*!
            ensures
                - print_status() assumes it is called with values which are linearly
                  approaching target().  It will display the current progress and attempt
                  to predict how much time is remaining until cur becomes equal to target().
                - prints a status message to out which indicates how much more time is
                  left until cur is equal to target()
                - if (always_print) then
                    - This function prints to the screen each time it is called.
                - else
                    - This function throttles the printing so that at most 1 message is
                      printed each second.  Note that it won't print anything to the screen
                      until about one second has elapsed.  This means that the first call
                      to print_status() never prints to the screen.
                - This function returns true if it prints to the screen and false
                  otherwise.
        !*/

        inline void finish (
            std::ostream& out = std::cout
        ) const;
        /*!
            ensures
                - This object prints the completed progress and the elapsed time to out.
                  It is meant to be called after the loop we are tracking the progress of.
        !*/

    private:

        double target_val;
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        double first_val;
        double seen_first_val;
        std::chrono::time_point<std::chrono::steady_clock> last_time;

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                               IMPLEMENTATION DETAILS
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    console_progress_indicator::
    console_progress_indicator (
        double target_value 
    ) :
        target_val(target_value),
        start_time(std::chrono::steady_clock::now()),
        first_val(0),
        seen_first_val(false),
        last_time(std::chrono::steady_clock::now())
    {
    }

// ----------------------------------------------------------------------------------------

    bool console_progress_indicator::
    print_status (
        double cur,
        bool always_print,
        std::ostream& out
    )
    {
        const auto cur_time = std::chrono::steady_clock::now();

        // if this is the first time print_status has been called
        // then collect some information and exit.  We will print status
        // on the next call.
        if (!seen_first_val)
        {
            start_time = cur_time;
            last_time = cur_time;
            first_val = cur;
            seen_first_val = true;
            return false;
        }

        if ((cur_time - last_time) >= std::chrono::seconds(1) || always_print)
        {
            last_time = cur_time;
            const auto delta_t = cur_time - start_time;
            double delta_val = std::abs(cur - first_val);

            // don't do anything if cur is equal to first_val
            if (delta_val < std::numeric_limits<double>::epsilon())
                return false;

            const auto rem_time = delta_t / delta_val * std::abs(target_val - cur);

            const auto oldflags = out.flags();
            out.setf(std::ios::fixed,std::ios::floatfield);
            std::streamsize ss;

            // adapt the precision based on whether the target val is an integer
            if (std::trunc(target_val) == target_val)
                ss = out.precision(0);
            else
                ss = out.precision(2);

            out << "Progress: " << cur << "/" << target_val;
            ss = out.precision(2);
            out << " (" << cur / target_val * 100. << "%). ";

            const auto hours = std::chrono::duration_cast<std::chrono::hours>(rem_time);
            const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(rem_time) - hours;
            const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(rem_time) - hours - minutes;
            out << "Time remaining: ";
            if (rem_time >= std::chrono::hours(1))
                out << hours.count() << "h ";
            if (rem_time >= std::chrono::minutes(1))
                out << minutes.count() << "min ";
            out << seconds.count() << "s.                \r" << std::flush;

            // restore previous output flags and precision settings
            out.flags(oldflags);
            out.precision(ss);

            return true;
        }

        return false;
    }

// ----------------------------------------------------------------------------------------

    double console_progress_indicator::
    target (
    ) const
    {
        return target_val;
    }

// ----------------------------------------------------------------------------------------

    void console_progress_indicator::
    reset (
        double target_value
    ) 
    {
        *this = console_progress_indicator(target_value);
    }

// ----------------------------------------------------------------------------------------

    void console_progress_indicator::
    finish (
        std::ostream& out
    ) const
    {
        const auto oldflags = out.flags();
        out.setf(std::ios::fixed,std::ios::floatfield);
        std::streamsize ss;

        // adapt the precision based on whether the target val is an integer
        if (std::trunc(target_val) == target_val)
            ss = out.precision(0);
        else
            ss = out.precision(2);

        out << "Progress: " << target_val << "/" << target_val;
        out << " (100.00%). ";
        const auto delta_t = std::chrono::steady_clock::now() - start_time;
        const auto hours = std::chrono::duration_cast<std::chrono::hours>(delta_t);
        const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(delta_t) - hours;
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(delta_t) - hours - minutes;
        out << "Time elapsed: ";
        if (delta_t >= std::chrono::hours(1))
            out << hours.count() << "h ";
        if (delta_t >= std::chrono::minutes(1))
            out << minutes.count() << "min ";
        out << seconds.count() << "s.                " << std::endl;

        // restore previous output flags and precision settings
        out.flags(oldflags);
        out.precision(ss);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CONSOLE_PROGRESS_INDiCATOR_Hh_

