// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SWIG_JAVA_ARRAY_H_
#define DLIB_SWIG_JAVA_ARRAY_H_


/*

    This file defines three special classes: array, array_view, and array_view_crit.  An
    array is a simple opaque handle to a java array, like a double[] array.  The array_view
    and array_view_crit objects allow you to access the contents of an array.  The
    interfaces of these objects is shown below, but for an example use, suppose you had an
    array of int in java and you wanted to pass it to C++.  You could create a C++ function
    like this:

        void my_function(const array_view<int32_t>& array);

    and then within java you could call it with code like this:

       int[] array = new int[100];
       my_function(array);

    and it will work just like you would expect.  The array_view<int32_t> will usually result in
    the JVM doing a copy in the background.  However, you can also declare your function
    like this:

        void my_function(const array_view_crit<int32_t>& array);

    and still call it the same way in java, however, using array_view_crit<int32_t> will usually
    not result in any copying, and is therefore very fast.  array_view_crit uses the JNI
    routine GetPrimitiveArrayCritical() to get a lock on the java memory underlying the
    array.  So it will probably prevent the garbage collector from running while your
    function is executing.  The JNI documentation is somewhat vague on the limitations of
    GetPrimitiveArrayCritical(), saying only that you shouldn't hold the lock on the array
    for "an extended period" or call back into the JVM.  Deciding whether or not this
    matters in your application is left as an exercise for the reader.


    There are two ways you can declare your methods if they take an array_view or
    array_view_crit.  Taking a const reference or a non-const reference.  E.g.
        void my_function(const array_view<int32_t>& array);
        void my_function(array_view<int32_t>& array);
    You can't declare them to be by value.  The non-const version allows you to modify the
    contents of the array and the modifications will be visible to java, as you would
    expect.  You can also make functions that take array objects directly, but that's only
    useful if you want to store the array handle somewhere, like in a member of a long
    lived class.  You can also write functions that return arrays back to java. E.g.
        array<int32_t> make_an_array(size_t s) 
        { 
            array<int32_t> arr(s);
            array_view<int32_t> aview(arr);
            // Use aview to put data into the array and generally do something useful. 
            ...
            return arr;
        }
    This would create an array and return it as a java int[] array.  


    You can also of course use functions taking many arguments, as is normally the case
    with SWIG.  Finally, these classes work with the following primitive types:
        - int16_t
        - int32_t
        - int64_t
        - char     (corresponding to java byte)
        - float
        - double




namespace java
{
    template <typename T>
    class array
    {
        /!*
            WHAT THIS OBJECT REPRESENTS
                This is a handle to a java array.  I.e. a reference to an array instance in
                java like a double[] or int[].  It doesn't do anything other than tell you
                the size of the array and allow you to hold a reference to it.

                To access the array contents, you need to create an array_view or
                array_view_crit from the array.
        *!/
    public:
        array();
        /!*
            ensures
                - #size() == 0
                - this array is a null reference, i.e. it doesn't reference any array.
        *!/

        explicit array(size_t new_size);
        /!*
            ensures
                - #size() == new_size
                - Allocates a new java array.
                - This array is a reference to the newly allocated java array object.
        *!/

        size_t size() const;
        /!*
            ensures
                - returns the number of elements in this java array.
        *!/

        void swap(array& item); 
        /!*
            ensures
                - swaps the state of *this and item.
        *!/

        array(const array& item);
        array& operator= (const array& item)
        array(array&& item);
        array& operator= (array&& item);
        /!*
            ensures
                - The array is copyable, assignable, and movable.  All copies will
                  reference the same underlying array.  So the copies are shallow, as is
                  normally the case with java reference semantics.
        *!/
    };



    template <typename T>
    class array_view
    {
        /!*
            WHAT THIS OBJECT REPRESENTS
                This is a view into a java array object.  It allows you to access the
                values stored in an array and modify them if you want to.

                You should only create array_view objects locally in a function since an
                array_view is only valid as long as the array it references exists.  So
                don't store array_view objects in the member area of a class or globally.
        *!/

    public:
        array_view();
        /!*
            ensures
                - #size() == 0
                - #data() == nullptr
        *!/

        array_view(const array<T>& arr, bool might_be_modified=true); 
        /!*
            ensures
                - #size() == arr.size()
                - #data() == a pointer to the beginning of the array data referenced by arr.
                - When you get a view on a java array, sometimes the JVM will actually
                  give you a pointer to a copy of the original array.  You therefore have
                  to tell the JVM if you modified the array when you are done using it.  If
                  you say you modified it then the JVM will perform another copy from your
                  memory buffer back into the JVM.  The state of might_be_modified controls
                  if we do this.  So if you are going to modify the array via this
                  array_view you should set might_be_modified==true.
        *!/

        size_t size() const; 
        /!*
            ensures
                - returns the number of elements in this java array.
        *!/

        T* data(); 
        const T* data() const; 
        /!*
            ensures
                - returns a pointer to the beginning of the array.  Or nullptr if this is a
                  handle to null, rather than an actual array instance.
        *!/

        T* begin(); 
        T* end(); 
        const T* begin() const; 
        const T* end() const; 
        /!*
            ensures
                - returns iterators to the start and one-past-the-end of the array, as is
                  the convention for iterator ranges in C++.
        *!/

        T& operator[](size_t i); 
        const T& operator[](size_t i) const; 
        /!*
            ensures
                - returns data()[i]
        *!/

    private:
        // this object is non-copyable.
        array_view(const array_view&);
        array_view& operator=(const array_view&);
    };


    template <typename T>
    class array_view_crit
    {
        /!*
            WHAT THIS OBJECT REPRESENTS
                This is just like an array_view and has an identical interface.  The only
                difference is that we use the JNI call GetPrimitiveArrayCritical() to get a
                critical lock on the array's memory.  Therefore, using array_view_crit is
                usually faster than array_view since it avoids any unnecessary copying back
                and forth between the JVM. 

                However, this critical lock can block the JVM's garbage collector from
                running.  So don't create long lived array_view_crit objects.
        *!/
    };

}
*/








// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                              IMPLEMENTATION DETAILS
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------








namespace java
{

template <typename T>
class array_view_base
{
public:
    array_view_base() = default;

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
    array_view_base(const array_view_base&);
    array_view_base& operator=(const array_view_base&);

};


// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template <typename T>
struct find_java_array_type;

template <> struct find_java_array_type<int16_t> { typedef jshortArray type; };
template <> struct find_java_array_type<int32_t> { typedef jintArray type; };
template <> struct find_java_array_type<int64_t> { typedef jlongArray type; };
template <> struct find_java_array_type<char>    { typedef jbyteArray type; };
template <> struct find_java_array_type<float>   { typedef jfloatArray type; };
template <> struct find_java_array_type<double>  { typedef jdoubleArray type; };

jshortArray  create_java_array(int16_t, size_t size) { return JNI_GetEnv()->NewShortArray(size); }
jintArray    create_java_array(int32_t, size_t size) { return JNI_GetEnv()->NewIntArray(size); }
jlongArray   create_java_array(int64_t, size_t size) { return JNI_GetEnv()->NewLongArray(size); }
jbyteArray   create_java_array(char,    size_t size) { return JNI_GetEnv()->NewByteArray(size); }
jfloatArray  create_java_array(float,   size_t size) { return JNI_GetEnv()->NewFloatArray(size); }
jdoubleArray create_java_array(double , size_t size) { return JNI_GetEnv()->NewDoubleArray(size); }

template <typename T>
class array
{
public:

    typedef typename find_java_array_type<T>::type java_type;

    array() {}

    explicit array(size_t size) 
    {
        ref = create_java_array(T(),size);
        is_global_ref = false;
    }

    array(java_type ref_)
    {
        if (ref_)
        {
            ref = (java_type)JNI_GetEnv()->NewGlobalRef(ref_);
            is_global_ref = true;
        }
    }

#ifndef SWIG
    array(array&& item)
    {
        ref = item.ref;
        is_global_ref = item.is_global_ref;
        item.ref = NULL;
        item.is_global_ref = false;
    }
    array& operator= (array&& item)
    {
        array(std::move(item)).swap(*this);
        return *this;
    }
#endif

    ~array()
    {
        if (ref)
        {
            // Don't delete the reference if it's a local reference, since the only reason
            // we will normally be using array object's that contain local references
            // is because we plan on returning the newly constructed array back to the JVM,
            // which automatically frees local references using the normal JVM garbage
            // collection scheme.
            if (is_global_ref)
                JNI_GetEnv()->DeleteGlobalRef(ref);

            ref = NULL;
            is_global_ref = false;
        }
    }

    size_t size() const 
    { 
        if (ref)
            return JNI_GetEnv()->GetArrayLength(ref); 
        else
            return 0;
    }

    array(const array& item)
    {
        array(item.ref).swap(*this);
    }

    array& operator= (const array& item)
    {
        array(item).swap(*this);
        return *this;
    }

    operator java_type() const { return ref;}

    void swap(array& item) { std::swap(ref, item.ref); }

private:
    java_type ref = NULL;
    bool is_global_ref = false;
};

#ifdef SWIG
// Tell SWIG to not use it's SwigValueWrapper stuff on array objects since they aren't
// needed and it causes superfluous construction and destruction of array objects.
%feature("novaluewrapper") array<int16_t>;
%template() array<int16_t>;
%feature("novaluewrapper") array<int32_t>;
%template() array<int32_t>;
%feature("novaluewrapper") array<int64_t>;
%template() array<int64_t>;
%feature("novaluewrapper") array<char>;
%template() array<char>;
%feature("novaluewrapper") array<float>;
%template() array<float>;
%feature("novaluewrapper") array<double>;
%template() array<double>;
#endif

#ifdef SWIG
%define tostring(token) 
    #token
%enddef

%define define_javaObjectRef_converion(type, java_type)
    // Define array conversions for non-const arrays
    %typemap(jtype)       (array<type>)  "java_type[]"
    %typemap(jstype)      (array<type>)  "java_type[]"
    %typemap(jni)         (array<type>)  tostring(j##java_type##Array)
    %typemap(javain)      (array<type>)  "$javainput"
    %typemap(in)          (array<type>)  { $1 = java::array<type>($input); }
    %typemap(javaout)     (array<type>)  {return $jnicall; }   
    %typemap(out)         (array<type>)  {jresult = result;}

    %typemap(jtype)       (array<type>&)  "java_type[]"
    %typemap(jstype)      (array<type>&)  "java_type[]"
    %typemap(jni)         (array<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (array<type>&)  "$javainput"
    %typemap(arginit)     (array<type>&) { $1 = &temp$argnum; }
    %typemap(in)          (array<type>&) (java::array<type> temp) { *($1) = java::array<type>($input); }

    %typemap(jtype)       (const array<type>&)  "java_type[]"
    %typemap(jstype)      (const array<type>&)  "java_type[]"
    %typemap(jni)         (const array<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (const array<type>&)  "$javainput"
    %typemap(arginit)     (const array<type>&) { $1 = &temp$argnum; }
    %typemap(in)          (const array<type>&) (java::array<type> temp) { *($1) = java::array<type>($input); }
%enddef
define_javaObjectRef_converion(int16_t,short)
define_javaObjectRef_converion(int32_t,int)
define_javaObjectRef_converion(int64_t,long)
define_javaObjectRef_converion(char,byte)
define_javaObjectRef_converion(float,float)
define_javaObjectRef_converion(double,double)

#endif
// ----------------------------------------------------------------------------------------

template <typename T> class array_view;

#define JAVA_ARRAY_CLASS_SPEC(ctype, type, Type)                                               \
template <> class array_view<ctype> : public array_view_base<ctype>                               \
{                                                                                           \
public:                                                                                     \
    ~array_view() { clear(); }                                                                 \
    array_view() {}                                                                            \
    array_view(const array<ctype>& arr, bool might_be_modified_=true){reset(JNI_GetEnv(),arr,might_be_modified_);} \
    void reset(JNIEnv* jenv_, j##type##Array arr, bool might_be_modified_) {                  \
        clear();                                                                            \
        jenv = jenv_;                                                                       \
        oldArr = arr;                                                                       \
        if (arr) {                                                                          \
            pdata = (ctype*)jenv->Get##Type##ArrayElements(arr, 0);                         \
            sz = jenv->GetArrayLength(arr);                                                 \
        }                                                                                   \
        might_be_modified = might_be_modified_;                                             \
    }                                                                                       \
private:                                                                                    \
    void clear() {                                                                          \
        if (pdata) {                                                                        \
            jenv->Release##Type##ArrayElements(oldArr, (j##type*)pdata, might_be_modified?0:JNI_ABORT); \
            pdata = nullptr;                                                                \
            sz = 0;                                                                         \
        }                                                                                   \
    }                                                                                       \
    JNIEnv* jenv = nullptr;                                                                 \
    j##type##Array oldArr;                                                                  \
    bool might_be_modified;                                                                   \
};

JAVA_ARRAY_CLASS_SPEC(int16_t,short, Short)
JAVA_ARRAY_CLASS_SPEC(int32_t,int, Int)
JAVA_ARRAY_CLASS_SPEC(int64_t,long, Long)
JAVA_ARRAY_CLASS_SPEC(char,byte, Byte)
JAVA_ARRAY_CLASS_SPEC(float,float, Float)
JAVA_ARRAY_CLASS_SPEC(double,double, Double)

 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------


template <typename T, typename JARR> 
class array_view_crit_base 
{
public:
    array_view_crit_base() = default;

    size_t size() const { return sz; }
    T* data() { return pdata; }
    const T* data() const { return pdata; }

    T* begin() { return pdata; }
    T* end() { return pdata+sz; }
    const T* begin() const { return pdata; }
    const T* end() const { return pdata+sz; }
    T& operator[](size_t i) { return pdata[i]; }
    const T& operator[](size_t i) const { return pdata[i]; }

    ~array_view_crit_base() { clear(); }

    void reset(JNIEnv* jenv_, JARR arr, bool might_be_modified_)
    {
        clear();
        jenv = jenv_;
        oldArr = arr;
        if (arr) 
        {
            pdata = (T*)jenv->GetPrimitiveArrayCritical(arr, 0);
            sz = jenv->GetArrayLength(arr);
        }
        might_be_modified = might_be_modified_;
    }

private:

    void clear()
    {
        if (pdata) {
            jenv->ReleasePrimitiveArrayCritical(oldArr, pdata, might_be_modified?0:JNI_ABORT);
            pdata = nullptr;
            sz = 0;
        }
    }

    // this object is non-copyable
    array_view_crit_base(const array_view_crit_base&);
    array_view_crit_base& operator=(const array_view_crit_base&);

    T* pdata = nullptr;
    size_t sz = 0;
    JNIEnv* jenv = nullptr;
    JARR oldArr;
    bool might_be_modified;
};

template <typename T> class array_view_crit;

template <> class array_view_crit<int16_t> : public array_view_crit_base<int16_t,jshortArray> { public: array_view_crit(){} array_view_crit(const array<int16_t>& arr, bool might_be_modified_=true){reset(JNI_GetEnv(),arr,might_be_modified_);} };
template <> class array_view_crit<int32_t> : public array_view_crit_base<int32_t,jintArray>   { public: array_view_crit(){} array_view_crit(const array<int32_t>& arr, bool might_be_modified_=true){reset(JNI_GetEnv(),arr,might_be_modified_);} };
template <> class array_view_crit<int64_t> : public array_view_crit_base<int64_t,jlongArray>  { public: array_view_crit(){} array_view_crit(const array<int64_t>& arr, bool might_be_modified_=true){reset(JNI_GetEnv(),arr,might_be_modified_);} };
template <> class array_view_crit<char>    : public array_view_crit_base<char,jbyteArray>     { public: array_view_crit(){} array_view_crit(const array<char>& arr, bool might_be_modified_=true){reset(JNI_GetEnv(),arr,might_be_modified_);} };
template <> class array_view_crit<float>   : public array_view_crit_base<float,jfloatArray>   { public: array_view_crit(){} array_view_crit(const array<float>& arr, bool might_be_modified_=true){reset(JNI_GetEnv(),arr,might_be_modified_);} };
template <> class array_view_crit<double>  : public array_view_crit_base<double,jdoubleArray> { public: array_view_crit(){} array_view_crit(const array<double>& arr, bool might_be_modified_=true){reset(JNI_GetEnv(),arr,might_be_modified_);} };

// ----------------------------------------------------------------------------------------

// Define SWIG typemaps so SWIG will know what to do with the array_view and array_view_crit
// objects.
#ifdef SWIG
%define define_array_converion(type, java_type)
    // Define array conversions for non-const arrays
    %typemap(jtype)       (array_view<type>&)  "java_type[]"
    %typemap(jstype)      (array_view<type>&)  "java_type[]"
    %typemap(jni)         (array_view<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (array_view<type>&)  "$javainput"
    %typemap(arginit)     (array_view<type>&)  { $1 = &temp$argnum; }
    %typemap(in)          (array_view<type>&) (java::array_view<type> temp)  { $1->reset(jenv, $input, true); }

    %typemap(jtype)       (const array_view<type>&)  "java_type[]"
    %typemap(jstype)      (const array_view<type>&)  "java_type[]"
    %typemap(jni)         (const array_view<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (const array_view<type>&)  "$javainput"
    %typemap(arginit)     (const array_view<type>&)  { $1 = &temp$argnum; }
    %typemap(in)          (const array_view<type>&) (java::array_view<type> temp)  { $1->reset(jenv, $input, false); }
%enddef
define_array_converion(int16_t,short)
define_array_converion(int32_t,int)
define_array_converion(int64_t,long)
define_array_converion(char,byte)
define_array_converion(float,float)
define_array_converion(double,double)



%define define_array_crit_converion(type, java_type)
    // Define array conversions for non-const arrays
    %typemap(jtype)       (array_view_crit<type>&)  "java_type[]"
    %typemap(jstype)      (array_view_crit<type>&)  "java_type[]"
    %typemap(jni)         (array_view_crit<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (array_view_crit<type>&)  "$javainput"
    %typemap(arginit)     (array_view_crit<type>&)  { $1 = &temp$argnum; }
    %typemap(in)          (array_view_crit<type>&) (java::array_view_crit<type> temp)  { $1->reset(jenv, $input, true); }

    %typemap(jtype)       (const array_view_crit<type>&)  "java_type[]"
    %typemap(jstype)      (const array_view_crit<type>&)  "java_type[]"
    %typemap(jni)         (const array_view_crit<type>&)  tostring(j##java_type##Array)
    %typemap(javain)      (const array_view_crit<type>&)  "$javainput"
    %typemap(arginit)     (const array_view_crit<type>&)  { $1 = &temp$argnum; }
    %typemap(in)          (const array_view_crit<type>&) (java::array_view_crit<type> temp)  { $1->reset(jenv, $input, false); }
%enddef
define_array_crit_converion(int16_t,short)
define_array_crit_converion(int32_t,int)
define_array_crit_converion(int64_t,long)
define_array_crit_converion(char,byte)
define_array_crit_converion(float,float)
define_array_crit_converion(double,double)

#endif // SWIG

}

#endif // DLIB_SWIG_JAVA_ARRAY_H_

