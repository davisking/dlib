#ifndef EXAMPLE_SWIG_ApI_H_ 
#define EXAMPLE_SWIG_ApI_H_

// This file is essentially a small unit test for the swig cmake scripts and the jvector
// classes.  All it does it define a few simple functions for writing to and summing
// arrays.  The swig_test.java file then calls these C++ functions and checks if they work
// correctly.  



// Let's use the jvector, a tool for efficiently binding java native arrays to C++ function
// arguments.  You do this by putting this pair of include statements in your swig_api.h
// file.  Then after that you can use the jvector and jvector_crit classes.  
#include "jvector.h"
#ifdef SWIG
%include "jvector.h"
#endif


// ----------------------------------------------------------------------------------------

// SWIG can't expose templated functions to java.  We declare these here as helper
// functions to make the non-templated routines swig will expose easier to write.  You can
// see these java exposed methods below (i.e. sum(), sum_crit(), assign(), and
// assign_crit()).
template <typename T>
T tsum(const jvector_crit<T>& arr)
{
    T s = 0;
    for (auto& v : arr)
        s += v;
    return s;
}
template <typename T>
T tsum(const jvector<T>& arr)
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

inline int sum_crit(const jvector_crit<int16_t>& arr) { return tsum(arr); }
inline int sum(const jvector<int16_t>& arr) { return tsum(arr); }
inline void assign_crit(jvector_crit<int16_t>& arr) { tassign(arr); }
inline void assign(jvector<int16_t>& arr) { tassign(arr); }


inline int sum_crit(const jvector_crit<int32_t>& arr) { return tsum(arr); }
inline int sum(const jvector<int32_t>& arr) { return tsum(arr); }
inline void assign_crit(jvector_crit<int32_t>& arr) { tassign(arr); }
inline void assign(jvector<int32_t>& arr) { tassign(arr); }


inline int sum_crit(const jvector_crit<int64_t>& arr) { return tsum(arr); }
inline int sum(const jvector<int64_t>& arr) { return tsum(arr); }
inline void assign_crit(jvector_crit<int64_t>& arr) { tassign(arr); }
inline void assign(jvector<int64_t>& arr) { tassign(arr); }


inline int sum_crit(const jvector_crit<char>& arr) { return tsum(arr); }
inline int sum(const jvector<char>& arr) { return tsum(arr); }
inline void assign_crit(jvector_crit<char>& arr) { tassign(arr); }
inline void assign(jvector<char>& arr) { tassign(arr); }



inline double sum_crit(const jvector_crit<double>& arr) { return tsum(arr); }
inline double sum(const jvector<double>& arr) { return tsum(arr); }
inline void assign_crit(jvector_crit<double>& arr) { tassign(arr); }
inline void assign(jvector<double>& arr) { tassign(arr); }


inline float sum_crit(const jvector_crit<float>& arr) { return tsum(arr); }
inline float sum(const jvector<float>& arr) { return tsum(arr); }
inline void assign_crit(jvector_crit<float>& arr) { tassign(arr); }
inline void assign(jvector<float>& arr) { tassign(arr); }


// ----------------------------------------------------------------------------------------


#endif // EXAMPLE_SWIG_ApI_H_


