#ifndef EXAMPLE_SWIG_ApI_H_ 
#define EXAMPLE_SWIG_ApI_H_

// This file is essentially a small unit test for the swig cmake scripts and the java array
// classes.  All it does it define a few simple functions for writing to and summing
// arrays.  The swig_test.java file then calls these C++ functions and checks if they work
// correctly.  



// Let's use java_array.h, a tool for efficiently binding java native arrays to C++
// function arguments.  You do this by putting this pair of include statements in your
// swig_api.h file.  Then after that you can use the java::array, java::array_view, and
// java::array_view_crit classes.  
#include <dlib/java/java_array.h>
#ifdef SWIG
%include <dlib/java/java_array.h>
#endif


using namespace java;


// SWIG can't expose templated functions to java.  We declare these here as helper
// functions to make the non-templated routines swig will expose easier to write.  You can
// see these java exposed methods below (i.e. sum(), sum_crit(), assign(), and
// assign_crit()).
template <typename T>
T tsum(const array_view_crit<T>& arr)
{
    T s = 0;
    for (auto& v : arr)
        s += v;
    return s;
}
template <typename T>
T tsum(const array_view<T>& arr)
{
    T s = 0;
    for (auto& v : arr)
        s += v;
    return s;
}
template <typename T>
void tassign(T& arr)
{
    for (size_t i = 0; i < arr.size(); ++i)
        arr[i] = i;
}

// ----------------------------------------------------------------------------------------

// Now write some functions SWIG will expose to java.  SWIG will automatically expose
// pretty much any non-template C++ code to java.  So just by defining these functions here
// we expose them to java.
// 
// All global C++ functions will appear in java as static member functions of class called
// "global", which is where these sum and assign routines will appear.  You can see
// examples of java code that calls them in swig_test.java.

inline int sum_crit(const array_view_crit<int16_t>& arr) { return tsum(arr); }
inline int sum(const array_view<int16_t>& arr) { return tsum(arr); }
inline void assign_crit(array_view_crit<int16_t>& arr) { tassign(arr); }
inline void assign(array_view<int16_t>& arr) { tassign(arr); }


inline int sum_crit(const array_view_crit<int32_t>& arr) { return tsum(arr); }
inline int sum(const array_view<int32_t>& arr) { return tsum(arr); }
inline void assign_crit(array_view_crit<int32_t>& arr) { tassign(arr); }
inline void assign(array_view<int32_t>& arr) { tassign(arr); }


inline int sum_crit(const array_view_crit<int64_t>& arr) { return tsum(arr); }
inline int sum(const array_view<int64_t>& arr) { return tsum(arr); }
inline void assign_crit(array_view_crit<int64_t>& arr) { tassign(arr); }
inline void assign(array_view<int64_t>& arr) { tassign(arr); }


inline int sum_crit(const array_view_crit<char>& arr) { return tsum(arr); }
inline int sum(const array_view<char>& arr) { return tsum(arr); }
inline void assign_crit(array_view_crit<char>& arr) { tassign(arr); }
inline void assign(array_view<char>& arr) { tassign(arr); }



inline double sum_crit(const array_view_crit<double>& arr) { return tsum(arr); }
inline double sum(const array_view<double>& arr) { return tsum(arr); }
inline void assign_crit(array_view_crit<double>& arr) { tassign(arr); }
inline void assign(array_view<double>& arr) { tassign(arr); }


inline float sum_crit(array<float> arr) 
{ 
    array_view_crit<float> a(arr);
    return tsum(a); 
}
inline float sum(const array<float>& arr) 
{ 
    array_view<float> a(arr);
    return tsum(a); 
}
inline void assign_crit(array_view_crit<float>& arr) { tassign(arr); }
inline void assign(array<float>& arr) 
{ 
    array_view<float> a(arr);
    tassign(a); 
}

array<int32_t> make_an_array(size_t s)
{
    array<int32_t> arr(s);
    array_view_crit<int32_t> a(arr);

    for (size_t i = 0; i < a.size(); ++i)
        a[i] = i;

    return arr;
}


// ----------------------------------------------------------------------------------------


#endif // EXAMPLE_SWIG_ApI_H_


