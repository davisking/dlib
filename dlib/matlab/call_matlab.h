// Copyright (C) 2012 Massachusetts Institute of Technology, Lincoln Laboratory
// License: Boost Software License   See LICENSE.txt for the full license.
// Authors: Davis E. King (davis@dlib.net)
#ifndef MIT_LL_CALL_MATLAB_H__
#define MIT_LL_CALL_MATLAB_H__

#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

void check_for_matlab_ctrl_c();
/*!
    ensures
        - If the user of MATLAB has pressed ctrl+c then this function will throw an
          exception.
!*/

// ----------------------------------------------------------------------------------------

class matlab_struct 
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            This object lets you interface with MATLAB structs from C++.  For example,
            given a MATLAB struct named mystruct, you could access it's fields like this:
                MATLAB way: mystruct.field
                C++ way:    mystruct["field"]
                MATLAB way: mystruct.field.subfield
                C++ way:    mystruct["field"]["subfield"]

            To get the values as C++ types you do something like this:
                int val = mystruct["field"];
            or 
                int val;  
                mystruct["field"].get(val);

            See also example_mex_struct.cpp for an example that uses this part of the API.
    !*/

    class sub;
public:
    matlab_struct() : struct_handle(0),should_free(false) {}
    ~matlab_struct();

    const sub operator[] (const std::string& name) const;
    sub operator[] (const std::string& name);
    bool has_field(const std::string& name) const;

    const void* release_struct_to_matlab() { const void* temp=struct_handle; struct_handle = 0; return temp; }
    void set_struct_handle(const void* sh) { struct_handle = sh; }
private:

    class sub 
    {
    public:
        sub() : struct_handle(0), field_idx(-1) {}

        template <typename T> operator T() const;
        template <typename T> void get(T& item) const; 
        template <typename T> sub& operator= (const T& new_val);
        const sub operator[] (const std::string& name) const;
        sub operator[] (const std::string& name);
        bool has_field(const std::string& name) const;
    private:
        friend class matlab_struct;
        const void* struct_handle;
        int field_idx;
        sub& operator=(const sub&);
    };
    const void* struct_handle;
    bool should_free;
    matlab_struct& operator=(const matlab_struct&); 
};

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template <typename T> 
struct output_decorator
{
    output_decorator(T& item_):item(item_){}
    T& item;
};

template <typename T>
output_decorator<T> returns(T& item) { return output_decorator<T>(item); }
/*!
    ensures
        - decorates item as an output type.  This stuff is used by the call_matlab()
          functions to tell if an argument is an input to the function or is supposed
          to be bound to one of the return arguments.
!*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

struct function_handle
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            This type is used to represent function handles passed from MATLAB into a
            mex function.  You can call the function referenced by the handle by
            saying:
                call_matlab(my_handle);
    !*/

    // These two lines are just implementation details, ignore them.
    function_handle():h(0){}
    void* const h;
};

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

void call_matlab (
    const std::string& function_name
);
/*!
    ensures
        - Calls MATLAB's function of the given name
!*/

// ----------------------------------------------------------------------------------------

void call_matlab (
    const function_handle& funct 
);
/*!
    ensures
        - Calls MATLAB's function represented by the handle funct
!*/

// ----------------------------------------------------------------------------------------

template <
    typename T1
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1
);
/*!
    ensures
        - calls MATLAB's function of the given name.  
        - if (A1 is not decorated as an output by returns()) then
            - A1 is passed as an argument into the MATLAB function
        - else
            - A1 is treated as the first return value from the MATLAB function.
!*/

template <
    typename T1
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1
) { call_matlab("feval", funct, A1); }
/*!
    ensures
        - Calls MATLAB's function represented by the handle funct
        - if (A1 is not decorated as an output by returns()) then
            - A1 is passed as an argument into the MATLAB function
        - else
            - A1 is treated as the first return value from the MATLAB function.
!*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
/*
    The rest of this file is just overloads of call_matlab() for up to 10 arguments (or
    just 9 arguments if function_handle is used).  They all do the same thing as the above 
    version of call_matlab().  Generally, any argument not decorated by returns() is an 
    input to the MATLAB function.  On the other hand, all arguments decorated by returns() 
    are treated as outputs.  
*/
// ----------------------------------------------------------------------------------------

template <
    typename T1, 
    typename T2
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1,
    const T2& A2
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, 
    typename T2,
    typename T3
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1,
    const T2& A2,
    const T3& A3
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, 
    typename T2,
    typename T3,
    typename T4
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, 
    typename T2,
    typename T3,
    typename T4,
    typename T5
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4,
    const T5& A5
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, 
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4,
    const T5& A5,
    const T6& A6
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, 
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4,
    const T5& A5,
    const T6& A6,
    const T7& A7
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, 
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7,
    typename T8
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4,
    const T5& A5,
    const T6& A6,
    const T7& A7,
    const T8& A8
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12& A12
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16, typename T17
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const T17& A17
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16, typename T17, typename T18
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const T17& A17,
    const T18& A18
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16, typename T17, typename T18, typename T19
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const T17& A17,
    const T18& A18, const T19& A19
);

// ----------------------------------------------------------------------------------------

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16, typename T17, typename T18, typename T19,
    typename T20
    >
void call_matlab (
    const std::string& function_name,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const T17& A17,
    const T18& A18, const T19& A19, const T20& A20
);

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template <
    typename T1,
    typename T2
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1,
    const T2& A2
)
{
    call_matlab("feval", funct, A1, A2);
}

template <
    typename T1,
    typename T2,
    typename T3
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1,
    const T2& A2,
    const T3& A3
)
{
    call_matlab("feval", funct, A1, A2, A3);
}

template <
    typename T1,
    typename T2,
    typename T3,
    typename T4
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4
)
{
    call_matlab("feval", funct, A1, A2, A3, A4);
}

template <
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4,
    const T5& A5
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5);
}

template <
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4,
    const T5& A5,
    const T6& A6
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6);
}

template <
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4,
    const T5& A5,
    const T6& A6,
    const T7& A7
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7);
}

template <
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7,
    typename T8
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1,
    const T2& A2,
    const T3& A3,
    const T4& A4,
    const T5& A5,
    const T6& A6,
    const T7& A7,
    const T8& A8
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16, typename T17
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const T17& A17
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16, typename T17, typename T18
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const T17& A17,
    const T18& A18
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18);
}

template <
    typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename
    T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16, typename T17, typename T18, typename T19
    >
void call_matlab (
    const function_handle& funct,
    const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
    const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const T12&
    A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const T17& A17,
    const T18& A18, const T19& A19
)
{
    call_matlab("feval", funct, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19);
}

// ----------------------------------------------------------------------------------------

// We define this function here so that, if you write some code that has check_for_matlab_ctrl_c()
// sprinkled throughout it you can still compile that code outside the mex wrapper
// environment and these calls will simply be no-ops.
#ifndef MATLAB_MEX_FILE
inline void check_for_matlab_ctrl_c() {}
#endif

}

#endif // MIT_LL_CALL_MATLAB_H__

