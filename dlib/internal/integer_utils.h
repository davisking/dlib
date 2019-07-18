// Copyright (C) 2019 Paul Dreik (github@pauldreik.se)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_INTEGER_UTILS
#define DLIB_INTEGER_UTILS
namespace dlib
{
namespace detail {
/**
 * @brief safe_multiply
 * multiplies a and b, without integer overflow.
 * @param a
 * @param b
 * @return a*b if possible, otherwise throws an exception
 */
long safe_multiply(long a, long b);
}
} // namespace dlib


#endif // DLIB_INTEGER_UTILS

