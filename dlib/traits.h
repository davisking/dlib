// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_TYPE_TRAITS_H
#define DLIB_TYPE_TRAITS_H

#include <numeric>
#include <complex>
#include <type_traits>
#include "enable_if.h"
#include "uintn.h"

namespace dlib
{
// ----------------------------------------------------------------------------------------
    /*!
        This is exactly the same as std::void_t.
    !*/
    
    #if __cplusplus < 201703L
    template<typename...> using void_t = void;
    #else
    using std::void_t;
    #endif

// ----------------------------------------------------------------------------------------

    /*!A is_complex
        This is a template that can be used to determine if a type is a specialization
        of the std::complex template class.

        For example:
            is_complex<float>::value == false              
            is_complex<std::complex<float> >::value == true   
    !*/

    template<typename T>
    struct is_complex : std::false_type {};
    template <typename T> 
    struct is_complex<std::complex<T> > : std::true_type {};
    template <typename T>
    struct is_complex<std::complex<T>& > : std::true_type {};
    template <typename T>
    struct is_complex<const std::complex<T>& > : std::true_type {};
    template <typename T>
    struct is_complex<const std::complex<T> > : std::true_type {};

// ----------------------------------------------------------------------------------------

    /*!A remove_complex
        This is a template that can be used to remove std::complex from the underlying type.

        For example:
            remove_complex<float>::type == float
            remove_complex<std::complex<float> >::type == float
    !*/
    template <typename T>
    struct remove_complex {typedef T type;};
    template <typename T>
    struct remove_complex<std::complex<T> > {typedef T type;};
    
    template<typename T>
    using remove_complex_t = typename remove_complex<T>::type;

// ----------------------------------------------------------------------------------------

    /*!A add_complex
        This is a template that can be used to add std::complex to the underlying type if it isn't already complex.

        For example:
            add_complex<float>::type == std::complex<float>
            add_complex<std::complex<float> >::type == std::complex<float>
    !*/
    template <typename T>
    struct add_complex {typedef std::complex<T> type;};
    template <typename T>
    struct add_complex<std::complex<T> > {typedef std::complex<T> type;};
    
    template<typename T>
    using add_complex_t = typename add_complex<T>::type;

// ----------------------------------------------------------------------------------------

    /*!A is_pointer_type

        This is a template where is_pointer_type<T>::value == true when T is a pointer 
        type and false otherwise.
    !*/

    template<typename T>
    using is_pointer_type = std::is_pointer<T>;
    
// ----------------------------------------------------------------------------------------

    /*!A is_const_type

        This is a template where is_const_type<T>::value == true when T is a const 
        type and false otherwise.
    !*/

    template <typename T>
    using is_const_type = std::is_const<T>;
    
// ----------------------------------------------------------------------------------------

    /*!A is_reference_type 

        This is a template where is_reference_type<T>::value == true when T is a reference 
        type and false otherwise.
    !*/

    template <typename T>
    using is_reference_type = std::is_reference<T>;

// ----------------------------------------------------------------------------------------

    /*!A is_same_type 

        This is a template where is_same_type<T,U>::value == true when T and U are the
        same type and false otherwise.   
    !*/

    template <typename T, typename U>
    using is_same_type = std::is_same<T,U>;
    
// ----------------------------------------------------------------------------------------

    /*!A is_float_type

        This is a template that can be used to determine if a type is one of the built
        int floating point types (i.e. float, double, or long double).
    !*/

    template <typename T> 
    using is_float_type = std::is_floating_point<T>;
    
// ----------------------------------------------------------------------------------------

    /*!A is_unsigned_type 

        This is a template where is_unsigned_type<T>::value == true when T is an unsigned
        scalar type and false when T is a signed scalar type.
    !*/
    template <typename T>
    using is_unsigned_type = std::is_unsigned<T>;

// ----------------------------------------------------------------------------------------

    /*!A is_signed_type 

        This is a template where is_signed_type<T>::value == true when T is a signed
        scalar type and false when T is an unsigned scalar type.
    !*/
    template <typename T>
    using is_signed_type = std::is_signed<T>;
    
// ----------------------------------------------------------------------------------------
    /*!A is_built_in_scalar_type
        
        This is a template that allows you to determine if the given type is a built
        in scalar type such as an int, char, float, short, etc.

        For example, is_built_in_scalar_type<char>::value == true
        For example, is_built_in_scalar_type<std::string>::value == false 
    !*/

    template <typename T> 
    using is_built_in_scalar_type = std::is_arithmetic<T>;
   
// ----------------------------------------------------------------------------------------

    /*!A basic_type

        This is a template that takes a type and strips off any const, volatile, or reference
        qualifiers and gives you back the basic underlying type. So for example:

        basic_type<const int&>::type == int
        basic_type<volatile const int&&>::type == int
    !*/

    template <typename T> 
    using basic_type = std::remove_cv<typename std::remove_reference<T>::type>;
  
// ----------------------------------------------------------------------------------------

    /*!A is_function 
        
        This is a template that allows you to determine if the given type is a function.

        For example,
            void funct();

            is_function<funct>::value == true
            is_function<int>::value == false 
    !*/

    template <typename T> 
    using is_function = std::is_function<T>;
        
// ----------------------------------------------------------------------------------------

    /*!A is_convertible

        This is a template that can be used to determine if one type is convertible 
        into another type.

        For example:
            is_convertible<int,float>::value == true    // because ints are convertible to floats
            is_convertible<int*,float>::value == false  // because int pointers are NOT convertible to floats
    !*/

    template <typename from, typename to>
    struct is_convertible
    {
        struct yes_type { char a; };
        struct no_type { yes_type a[2]; };
        static const from& from_helper();
        static yes_type test(to);
        static no_type test(...);
        const static bool value = sizeof(test(from_helper())) == sizeof(yes_type);
    };
    
// ----------------------------------------------------------------------------------------

    /*!A promote 
        
        This is a template that takes one of the built in scalar types and gives you another
        scalar type that should be big enough to hold sums of values from the original scalar 
        type.  The new scalar type will also always be signed.

        For example, promote<uint16>::type == int32
    !*/

    template <typename T, size_t s = sizeof(T)> struct promote;
    template <typename T> struct promote<T,1> { typedef int32 type; };
    template <typename T> struct promote<T,2> { typedef int32 type; };
    template <typename T> struct promote<T,4> { typedef int64 type; };
    template <typename T> struct promote<T,8> { typedef int64 type; };

    template <> struct promote<float,sizeof(float)>             { typedef double type; };
    template <> struct promote<double,sizeof(double)>           { typedef double type; };
    template <> struct promote<long double,sizeof(long double)> { typedef long double type; };
    
// ----------------------------------------------------------------------------------------
    
    /*!A is_std_hashable
        This is a type trait that determines if there exists a std::hash 
        specialisation for type T

        For example:
            is_std_hashable<std::string>::value == true
    !*/
    
    template <typename T, typename = void>
    struct is_std_hashable : std::false_type {};

    template <typename T>
    struct is_std_hashable<T, void_t<decltype(std::declval<std::hash<T>>()(std::declval<const T&>())),
                                     decltype(std::declval<std::hash<T>>()(std::declval<T const&>()))>> : std::true_type {};
                
// ----------------------------------------------------------------------------------------
                                     
    namespace swap_details
    {
        using std::swap;

        template <typename T, typename = void>
        struct is_swappable : std::false_type {};

        template <typename T>
        struct is_swappable<T, void_t<decltype(swap(std::declval<T&>(), std::declval<T&>()))>> : std::true_type {};
    }
    
    /*!A is_swappable
        This is a type trait that determines if a type can be used with swap()

        For example:
            is_swappable<std::string>::value == true
    !*/
    
    template <typename T>
    struct is_swappable : public swap_details::is_swappable<T> {};
    
// ----------------------------------------------------------------------------------------   
}

#endif //DLIB_TYPE_TRAITS_H