// Copyright (C) 2019 Paul Dreik (github@pauldreik.se)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "integer_utils.h"

#include <stdexcept>
#include <cstdint>
#include <limits>

namespace dlib
{
namespace detail {
long safe_multiply(long a, long b) {
    long ret;
    bool overflow=false;
#ifdef DLIB_HAS_BUILTIN_MUL_OVERFLOW
    // works for clang and gcc
    overflow=__builtin_mul_overflow(a,b,&ret);
#else
    using small=long;
    using large=std::intmax_t;
    if(sizeof(small)<sizeof(large)) {
        // fallback: promoting to a larger type
        // This works for MSVC (yet only tested on compiler explorer).
        auto tmp=large{a}*b;
        if(tmp<=std::numeric_limits<small>::max() && tmp>=std::numeric_limits<small>::min()) {
            overflow=false;
            ret=tmp;
        } else {
            overflow=true;
        }
    } else {
       // unsupported compiler
       throw std::logic_error("signed integer overflow check not implemented on this platform");
    }
#endif
    if(overflow) {
        throw std::runtime_error("signed integer overflow avoided");
    }
    return ret;
}
}
} // namespace dlib
