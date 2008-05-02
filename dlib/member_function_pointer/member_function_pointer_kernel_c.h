// Copyright (C) 2005  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MEMBER_FUNCTION_POINTER_KERNEl_C_
#define DLIB_MEMBER_FUNCTION_POINTER_KERNEl_C_

#include "member_function_pointer_kernel_abstract.h"
#include "../algs.h"
#include "../assert.h"

namespace dlib
{

    template <
        typename mfpb,// is an implementation of member_function_pointer_kernel_abstract.h
        typename PARAM1 = typename mfpb::param1_type,
        typename PARAM2 = typename mfpb::param2_type,
        typename PARAM3 = typename mfpb::param3_type,
        typename PARAM4 = typename mfpb::param4_type 
        >
    class member_function_pointer_kernel_c;

    template <
        typename mfpb,
        typename PARAM1,
        typename PARAM2,
        typename PARAM3,
        typename PARAM4
        >
    void swap (
        member_function_pointer_kernel_c<mfpb,PARAM1,PARAM2,PARAM3,PARAM4>& a,
        member_function_pointer_kernel_c<mfpb,PARAM1,PARAM2,PARAM3,PARAM4>& b
    ) { a.swap(b); }

// ----------------------------------------------------------------------------------------

    template <
        typename mfpb
        >
    class member_function_pointer_kernel_c<mfpb,void,void,void,void> : 
    public mfpb
    {
    public:

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)()
        )
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(cb != 0,
                   "\tvoid member_function_pointer::set"
                   << "\n\tthe member function pointer can't be null"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::set(object,cb);
        }

        void operator () (
        ) const
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(this->is_set() == true ,
                   "\tvoid member_function_pointer::operator()"
                   << "\n\tYou must call set() before you can use this function"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::operator()();
        }
    };

// ----------------------------------------------------------------------------------------

    template <
        typename mfpb,
        typename PARAM1
        >
    class member_function_pointer_kernel_c<mfpb,PARAM1,void,void,void> : 
    public mfpb
    {
    public:

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)(PARAM1)
        )
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(cb != 0,
                   "\tvoid member_function_pointer::set"
                   << "\n\tthe member function pointer can't be null"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::set(object,cb);
        }

        void operator () (
            PARAM1 param1
        ) const
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(this->is_set() == true ,
                   "\tvoid member_function_pointer::operator()"
                   << "\n\tYou must call set() before you can use this function"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::operator()(param1);
        }
    };

// ----------------------------------------------------------------------------------------

    template <
        typename mfpb,
        typename PARAM1,
        typename PARAM2
        >
    class member_function_pointer_kernel_c<mfpb,PARAM1,PARAM2,void,void> : 
    public mfpb
    {
    public:

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)(PARAM1,PARAM2)
        )
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(cb != 0,
                   "\tvoid member_function_pointer::set"
                   << "\n\tthe member function pointer can't be null"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::set(object,cb);
        }

        void operator () (
            PARAM1 param1,
            PARAM2 param2
        ) const
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(this->is_set() == true ,
                   "\tvoid member_function_pointer::operator()"
                   << "\n\tYou must call set() before you can use this function"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::operator()(param1,param2);
        }
    };

// ----------------------------------------------------------------------------------------

    template <
        typename mfpb,
        typename PARAM1,
        typename PARAM2,
        typename PARAM3
        >
    class member_function_pointer_kernel_c<mfpb,PARAM1,PARAM2,PARAM3,void> : 
    public mfpb
    {
    public:

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)(PARAM1,PARAM2,PARAM3)
        )
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(cb != 0,
                   "\tvoid member_function_pointer::set"
                   << "\n\tthe member function pointer can't be null"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::set(object,cb);
        }

        void operator () (
            PARAM1 param1,
            PARAM2 param2,
            PARAM3 param3
        ) const
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(this->is_set() == true ,
                   "\tvoid member_function_pointer::operator()"
                   << "\n\tYou must call set() before you can use this function"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::operator()(param1,param2,param3);
        }
    };

// ----------------------------------------------------------------------------------------

    template <
        typename mfpb,
        typename PARAM1,
        typename PARAM2,
        typename PARAM3,
        typename PARAM4
        >
    class member_function_pointer_kernel_c : 
    public mfpb
    {
    public:

        template <
            typename T
            >
        void set (
            T& object,
            void (T::*cb)(PARAM1,PARAM2,PARAM3,PARAM4)
        )
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(cb != 0,
                   "\tvoid member_function_pointer::set"
                   << "\n\tthe member function pointer can't be null"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::set(object,cb);
        }

        void operator () (
            PARAM1 param1,
            PARAM2 param2,
            PARAM3 param3,
            PARAM4 param4
        ) const
        {
            // make sure requires clause is not broken
            DLIB_CASSERT(this->is_set() == true ,
                   "\tvoid member_function_pointer::operator()"
                   << "\n\tYou must call set() before you can use this function"
                   << "\n\tthis: " << this
            );

            // call the real function
            mfpb::operator()(param1,param2,param3,param4);
        }
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MEMBER_FUNCTION_POINTER_KERNEl_C_

