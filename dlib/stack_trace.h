// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STACK_TRACe_
#define DLIB_STACK_TRACe_

/*!
    This file defines 3 things.  Two of them are preprocessor macros that
    enable you to tag functions with the dlib stack trace watcher.  The
    third thing is a function named get_stack_trace() which returns the
    current stack trace in std::string form.

    To enable the stack trace you must #define DLIB_ENABLE_STACK_TRACE.
    When this #define isn't set then the 3 things described above
    still exist but they don't do anything.

    Also note that when the stack trace is enabled it changes the DLIB_ASSERT
    and DLIB_CASSERT macros so that they print stack traces when 
    an assert fails.

    See the following example program for details:

    #include <iostream>
    #include <dlib/stack_trace.h>

    void funct2()
    {
        // put this macro at the top of each function you would
        // like to appear in stack traces
        DLIB_STACK_TRACE;

        // you may print the current stack trace as follows. 
        std::cout << get_stack_trace() << endl;
    }

    void funct()
    {
        // This alternate form of DLIB_STACK_TRACE allows you to specify
        // the string used to name the current function.  The other form
        // will usually output an appropriate function name automatically
        // so this may not be needed.
        DLIB_STACK_TRACE_NAMED("funct");
        funct2();
    }

    int main()
    {
        funct();
    }
!*/


#include <sstream>
#include <cstring>
#include <string>
#include "assert.h"

// only setup the stack trace stuff if the asserts are enabled (which happens in debug mode
// basically)
#ifdef DLIB_ENABLE_STACK_TRACE 

namespace dlib
{
    inline const std::string get_stack_trace();
}

// redefine the DLIB_CASSERT macro to include the stack trace
#undef DLIB_CASSERT
#define DLIB_CASSERT(_exp,_message)                                              \
    {if ( !(_exp) )                                                         \
    {                                                                       \
        std::ostringstream dlib__out;                                       \
        dlib__out << "\n\nError occurred at line " << __LINE__ << ".\n";    \
        dlib__out << "Error occurred in file " << __FILE__ << ".\n";      \
        dlib__out << "Error occurred in function " << DLIB_FUNCTION_NAME << ".\n\n";      \
        dlib__out << "Failing expression was " << #_exp << ".\n";           \
        dlib__out << _message << "\n\n";                                      \
        dlib__out << "Stack Trace: \n" << get_stack_trace() << "\n";        \
        dlib_assert_breakpoint();                                           \
        throw dlib::fatal_error(dlib::EBROKEN_ASSERT,dlib__out.str());      \
    }}                                                                      


#include "threads.h"
#include "stack.h"

namespace dlib
{

    class stack_tracer
    {
    public:
        stack_tracer (
            const char* funct_name,
            const char* file_name,
            const int line_number
        )
        {
            std::ostringstream sout;
            // if the function name isn't really long then put this all on a single line
            if (std::strlen(funct_name) < 40)
                sout << file_name << ":" << line_number << ": " << funct_name;
            else
                sout << file_name << ":" << line_number << "\n" << funct_name;

            // pop the string onto the function stack trace
            std::string temp(sout.str());
            trace().data().push(temp);
        }

        ~stack_tracer()
        {
            std::string temp;
            trace().data().pop(temp);
        }

        static const std::string get_stack_trace()
        {
            std::ostringstream sout;
            trace().data().reset();
            while (trace().data().move_next())
            {
                sout << trace().data().element() << "\n";
            }
            return sout.str();
        }

    private:

        typedef stack<std::string>::kernel_1a_c stack_of_string;
        static thread_specific_data<stack_of_string>& trace()
        {
            static thread_specific_data<stack_of_string> a;
            return a;
        }
    };

    inline const std::string get_stack_trace()
    {
        return stack_tracer::get_stack_trace();
    }

}

#define DLIB_STACK_TRACE_NAMED(x) stack_tracer dlib_stack_tracer_object(x,__FILE__,__LINE__)
#define DLIB_STACK_TRACE stack_tracer dlib_stack_tracer_object(DLIB_FUNCTION_NAME,__FILE__,__LINE__)

#else // don't do anything if ENABLE_ASSERTS isn't defined
#define DLIB_STACK_TRACE_NAMED(x) 
#define DLIB_STACK_TRACE 

namespace dlib
{
    inline const std::string get_stack_trace() { return std::string("stack trace not enabled");}
}

#endif


#endif // DLIB_STACK_TRACe_

