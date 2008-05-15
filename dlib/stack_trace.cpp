// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STACK_TRACE_CPp_
#define DLIB_STACK_TRACE_CPp_

#if defined(DLIB_ENABLE_STACK_TRACE) && !defined(NO_MAKEFILE)

#include <sstream>
#include <cstring>
#include "stack_trace.h"
#include "threads.h"
#include "stack.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    thread_specific_data<stack<std::string>::kernel_1a>& get_dlib_stack_trace_stack()
    {
        static thread_specific_data<stack<std::string>::kernel_1a> a;
        return a;
    }

// ----------------------------------------------------------------------------------------

    stack_tracer::
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
        get_dlib_stack_trace_stack().data().push(temp);
    }

// ----------------------------------------------------------------------------------------

    stack_tracer::
    ~stack_tracer()
    {
        std::string temp;
        get_dlib_stack_trace_stack().data().pop(temp);
    }

// ----------------------------------------------------------------------------------------

    const std::string get_stack_trace()
    {
        std::ostringstream sout;
        get_dlib_stack_trace_stack().data().reset();
        while (get_dlib_stack_trace_stack().data().move_next())
        {
            sout << get_dlib_stack_trace_stack().data().element() << "\n";
        }
        return sout.str();
    }

// ----------------------------------------------------------------------------------------

}
#endif

#endif // DLIB_STACK_TRACE_CPp_


