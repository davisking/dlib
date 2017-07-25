// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SWIG_JVECTOR_H_
#define DLIB_SWIG_JVECTOR_H_


/*

    This file defines two special classes, jvector and jvector_crit.  Both of them have the
    interface defined by jvector_base, that is, the interface of a simple array object like
    std::vector (except without any ability to be resized).  These classes are simple
    interfaces to java native arrays.  So for example, suppose you had an array of int in
    java and you wanted to pass it to C++.  You could create a C++ function like this:

        void my_function(const jvector<int>& array);

    and then within java you could call it with code like this:

       int[] array = new int[100];
       my_function(array);

    and it will work just like you would expect.  The jvector<int> will usually result in
    the JVM doing a copy in the background.  However, you can also declare your function
    like this:

        void my_function(const jvector_crit<int>& array);

    and still call it the same way in java, however, using jvector_crit<int> will usually
    not result in any copying, and is therefore very fast.  jvector_crit uses the JNI
    routine GetPrimitiveArrayCritical() to get a lock on the java memory underlying the
    array.  So it will probably prevent the garbage collector from running while your
    function is executing.  The JNI documentation is somewhat vague on the limitations of
    GetPrimitiveArrayCritical(), saying only that you shouldn't hold the lock on the array
    for "an extended period" or call back into the JVM.  Deciding whether or not this
    matters in your application is left as an exercise for the reader.


    There are two ways you can declare your methods.  Taking a const reference or a
    non-const reference.  E.g.:
        void my_function(const jvector<int>& array);
        void my_function(jvector<int>& array);
    The non-const version allows you to modify the contents of the array and the
    modifications will be visible to java, as you would expect.

    You can also of course use functions taking many arguments, as is normally the case
    with SWIG.  Finally, jvector works with the following primitive types:
        - int16_t
        - int32_t
        - int64_t
        - char     (corresponding to java byte)
        - float
        - double
*/


// ----------------------------------------------------------------------------------------

template <typename T>
class jvector_base
{
public:
    jvector_base() = default;

    size_t size() const { return sz; }
    T* data() { return pdata; }
    const T* data() const { return pdata; }

    T* begin() { return pdata; }
    T* end() { return pdata+sz; }
    const T* begin() const { return pdata; }
    const T* end() const { return pdata+sz; }

    T& operator[](size_t i) { return pdata[i]; }
    const T& operator[](size_t i) const { return pdata[i]; }

protected:
    T* pdata = nullptr;
    size_t sz = 0;

private:
    // this object is non-copyable
    jvector_base(const jvector_base&);
    jvector_base& operator=(const jvector_base&);

};


// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template <typename T> class jvector;

#define JVECTOR_CLASS_SPEC(ctype, type, Type)                                               \
template <> class jvector<ctype> : public jvector_base<ctype>                               \
{                                                                                           \
public:                                                                                     \
    ~jvector() { clear(); }                                                                 \
    void reset(JNIEnv* jenv_, j##type##Array arr, bool mightBeModified_) {                  \
        clear();                                                                            \
        jenv = jenv_;                                                                       \
        oldArr = arr;                                                                       \
        pdata = (ctype*)jenv->Get##Type##ArrayElements(arr, 0);                             \
        sz = jenv->GetArrayLength(arr);                                                     \
        mightBeModified = mightBeModified_;                                                 \
    }                                                                                       \
private:                                                                                    \
    void clear() {                                                                          \
        if (pdata) {                                                                        \
            jenv->Release##Type##ArrayElements(oldArr, (j##type*)pdata, mightBeModified?0:JNI_ABORT); \
            pdata = nullptr;                                                                \
        }                                                                                   \
    }                                                                                       \
    JNIEnv* jenv = nullptr;                                                                 \
    j##type##Array oldArr;                                                                  \
    bool mightBeModified;                                                                   \
};

JVECTOR_CLASS_SPEC(int16_t,short, Short)
JVECTOR_CLASS_SPEC(int32_t,int, Int)
JVECTOR_CLASS_SPEC(int64_t,long, Long)
JVECTOR_CLASS_SPEC(char,byte, Byte)
JVECTOR_CLASS_SPEC(float,float, Float)
JVECTOR_CLASS_SPEC(double,double, Double)

 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template <typename T, typename JARR> 
class jvector_crit_base 
{
public:
    jvector_crit_base() = default;

    size_t size() const { return sz; }
    T* data() { return pdata; }
    const T* data() const { return pdata; }

    T* begin() { return pdata; }
    T* end() { return pdata+sz; }
    const T* begin() const { return pdata; }
    const T* end() const { return pdata+sz; }
    T& operator[](size_t i) { return pdata[i]; }
    const T& operator[](size_t i) const { return pdata[i]; }

    ~jvector_crit_base() { clear(); }

    void reset(JNIEnv* jenv_, JARR arr, bool mightBeModified_)
    {
        clear();
        jenv = jenv_;
        oldArr = arr;
        pdata = (T*)jenv->GetPrimitiveArrayCritical(arr, 0);
        sz = jenv->GetArrayLength(arr);
        mightBeModified = mightBeModified_;
    }

private:

    void clear()
    {
        if (pdata) {
            jenv->ReleasePrimitiveArrayCritical(oldArr, pdata, mightBeModified?0:JNI_ABORT);
            pdata = nullptr;
        }
    }

    // this object is non-copyable
    jvector_crit_base(const jvector_crit_base&);
    jvector_crit_base& operator=(const jvector_crit_base&);

    T* pdata = nullptr;
    size_t sz = 0;
    JNIEnv* jenv = nullptr;
    JARR oldArr;
    bool mightBeModified;
};

template <typename T> class jvector_crit;

template <> class jvector_crit<int16_t> : public jvector_crit_base<int16_t,jshortArray> {};
template <> class jvector_crit<int32_t> : public jvector_crit_base<int32_t,jintArray> {};
template <> class jvector_crit<int64_t> : public jvector_crit_base<int64_t,jlongArray> {};
template <> class jvector_crit<char>    : public jvector_crit_base<char,jbyteArray> {};
template <> class jvector_crit<float>   : public jvector_crit_base<float,jfloatArray> {};
template <> class jvector_crit<double>  : public jvector_crit_base<double,jdoubleArray> {};
 
// ----------------------------------------------------------------------------------------

// Define SWIG typemaps so SWIG will know what to do with the jvector and jvector_crit
// objects.
#ifdef SWIG
%define tostring(token) 
    #token
%enddef

%define define_jvector_converion(type, java_type)
    // Define array conversions for non-const arrays
    %typemap(jtype)       (jvector<type>&)  "java_type[]"
    %typemap(jstype)      (jvector<type>&)  "java_type[]"
    %typemap(jni)         (jvector<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (jvector<type>&)  "$javainput"
    %typemap(arginit)     (jvector<type>&)  { $1 = &temp$argnum; }
    %typemap(in)          (jvector<type>&) (jvector<type> temp)  { $1->reset(jenv, $input, true); }

    %typemap(jtype)       (const jvector<type>&)  "java_type[]"
    %typemap(jstype)      (const jvector<type>&)  "java_type[]"
    %typemap(jni)         (const jvector<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (const jvector<type>&)  "$javainput"
    %typemap(arginit)     (const jvector<type>&)  { $1 = &temp$argnum; }
    %typemap(in)          (const jvector<type>&) (jvector<type> temp)  { $1->reset(jenv, $input, false); }
%enddef
define_jvector_converion(int16_t,short)
define_jvector_converion(int32_t,int)
define_jvector_converion(int64_t,long)
define_jvector_converion(char,byte)
define_jvector_converion(float,float)
define_jvector_converion(double,double)



%define define_jvector_crit_converion(type, java_type)
    // Define array conversions for non-const arrays
    %typemap(jtype)       (jvector_crit<type>&)  "java_type[]"
    %typemap(jstype)      (jvector_crit<type>&)  "java_type[]"
    %typemap(jni)         (jvector_crit<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (jvector_crit<type>&)  "$javainput"
    %typemap(arginit)     (jvector_crit<type>&)  { $1 = &temp$argnum; }
    %typemap(in)          (jvector_crit<type>&) (jvector_crit<type> temp)  { $1->reset(jenv, $input, true); }

    %typemap(jtype)       (const jvector_crit<type>&)  "java_type[]"
    %typemap(jstype)      (const jvector_crit<type>&)  "java_type[]"
    %typemap(jni)         (const jvector_crit<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (const jvector_crit<type>&)  "$javainput"
    %typemap(arginit)     (const jvector_crit<type>&)  { $1 = &temp$argnum; }
    %typemap(in)          (const jvector_crit<type>&) (jvector_crit<type> temp)  { $1->reset(jenv, $input, false); }
%enddef
define_jvector_crit_converion(int16_t,short)
define_jvector_crit_converion(int32_t,int)
define_jvector_crit_converion(int64_t,long)
define_jvector_crit_converion(char,byte)
define_jvector_crit_converion(float,float)
define_jvector_crit_converion(double,double)

#endif // SWIG

#endif // DLIB_SWIG_JVECTOR_H_

