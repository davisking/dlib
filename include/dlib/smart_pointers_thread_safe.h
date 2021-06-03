// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SMART_POINTERs_THREAD_SAFE_H_
#define DLIB_SMART_POINTERs_THREAD_SAFE_H_ 

// This is legacy smart pointer code that will likely to stop working under default
// compiler flags when C++17 becomes the default standard in the compilers.
// Please consider migrating your code to contemporary smart pointers from C++
// standard library. The warning below will help to detect if the deprecated code
// was included from library's clients.
#if (defined(__GNUC__) && ((__GNUC__ >= 4 && __GNUC_MINOR__ >= 8) || (__GNUC__ > 4))) || \
  (defined(__clang__) && ((__clang_major__ >= 3 && __clang_minor__ >= 4)))
#pragma GCC warning "smart_pointers_thread_safe.h is included which will fail to compile under C++17"
#endif

#include "smart_pointers/shared_ptr_thread_safe.h"

#endif // DLIB_SMART_POINTERs_THREAD_SAFE_H_ 



