// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TYPE_SAFE_UNIOn_h_ 
#define DLIB_TYPE_SAFE_UNIOn_h_

#include "type_safe_union_kernel_abstract.h"
#include "../algs.h"
#include "../noncopyable.h"
#include <new>

namespace dlib
{
// ----------------------------------------------------------------------------------------

    template <
        typename T1,
        typename T2 = T1,
        typename T3 = T1,
        typename T4 = T1,
        typename T5 = T1, 
        typename T6 = T1,
        typename T7 = T1,
        typename T8 = T1,
        typename T9 = T1,
        typename T10 = T1
        >
    class type_safe_union : noncopyable
    {
        /*!
            CONVENTION
                - is_empty() ==  (type_identity == 0)
                - contains<T>() == (type_identity == get_id<T>())
                - mem.data == the block of memory on the stack which is
                  where objects in the union are stored
        !*/

    private:

        const static size_t max_size = tmax<tmax<tmax<tmax<tmax<tmax<tmax<tmax<tmax<sizeof(T1),
                                                        sizeof(T2)>::value,
                                                        sizeof(T3)>::value,
                                                        sizeof(T4)>::value,
                                                        sizeof(T5)>::value,
                                                        sizeof(T6)>::value,
                                                        sizeof(T7)>::value,
                                                        sizeof(T8)>::value,
                                                        sizeof(T9)>::value,
                                                        sizeof(T10)>::value; 

        union mem_block
        {
            // All of this garbage is to make sure this union is properly aligned 
            // (a union is always aligned such that everything in it would be properly
            // aligned.  So the assumption here is that one of these objects has 
            // a large enough alignment requirement to satisfy any object this
            // type_safe_union might contain).
            void* void_ptr;
            struct {
                void (type_safe_union::*callback)();
                type_safe_union* o; 
            } stuff;
            long double more_stuff;

            char data[max_size]; 
        };


        int type_identity;
        mem_block mem;

        struct destruct_helper
        {
            template <typename T>
            void operator() (T& item) const
            {
                item.~T();
            }
        };

        void destruct (
        ) 
        /*!
            ensures
                - #is_empty() == true
        !*/
        {
            // destruct whatever is in this object
            apply_to_contents(destruct_helper());

            // mark this object as being empty
            type_identity = 0;
        }

        template <typename T>
        int get_id (
        ) const
        {
            if (is_same_type<T,T1>::value) return 1;
            if (is_same_type<T,T2>::value) return 2;
            if (is_same_type<T,T3>::value) return 3;
            if (is_same_type<T,T4>::value) return 4;
            if (is_same_type<T,T5>::value) return 5;

            if (is_same_type<T,T6>::value) return 6;
            if (is_same_type<T,T7>::value) return 7;
            if (is_same_type<T,T8>::value) return 8;
            if (is_same_type<T,T9>::value) return 9;
            if (is_same_type<T,T10>::value) return 10;

            // return a number that doesn't match any of the
            // valid states of type_identity
            return 10000;
        }

        template <typename T>
        void construct (
        )  
        { 
            if (type_identity != get_id<T>())
            {
                destruct(); 
                new((void*)mem.data) T(); 
                type_identity = get_id<T>();
            }
        }

        template <typename T>
        void operator() (T& item) 
        /*
            This function is used by the swap function of this class.  See that
            function to see how this works.
        */
        {
            exchange(get<T>(), item);
        }

    public:

        type_safe_union() : type_identity(0) 
        { 
        }

        ~type_safe_union()
        {
            destruct();
        }

        template <typename T>
        bool contains (
        ) const
        {
            return type_identity == get_id<T>();
        }

        bool is_empty (
        ) const
        {
            return type_identity == 0;
        }

        template <
            typename T
            >
        void apply_to_contents (
            T& obj
        ) 
        {
            switch (type_identity)
            {
                // do nothing because we are empty
                case 0: break;

                case 1: obj(get<T1>());  break;
                case 2: obj(get<T2>());  break;
                case 3: obj(get<T3>());  break;
                case 4: obj(get<T4>());  break;
                case 5: obj(get<T5>());  break;

                case 6: obj(get<T6>());  break;
                case 7: obj(get<T7>());  break;
                case 8: obj(get<T8>());  break;
                case 9: obj(get<T9>());  break;
                case 10: obj(get<T10>());  break;
            }
        }

        template <
            typename T
            >
        void apply_to_contents (
            const T& obj
        ) 
        {
            switch (type_identity)
            {
                // do nothing because we are empty
                case 0: break;

                case 1: obj(get<T1>());  break;
                case 2: obj(get<T2>());  break;
                case 3: obj(get<T3>());  break;
                case 4: obj(get<T4>());  break;
                case 5: obj(get<T5>());  break;

                case 6: obj(get<T6>());  break;
                case 7: obj(get<T7>());  break;
                case 8: obj(get<T8>());  break;
                case 9: obj(get<T9>());  break;
                case 10: obj(get<T10>());  break;
            }
        }

        void swap (
            type_safe_union& item
        )
        {
            // if both *this and item contain the same type of thing
            if (type_identity == item.type_identity)
            {
                // swap the things in this and item.  
                item.apply_to_contents(*this);
            }
            else if (type_identity == 0)
            {
                // *this doesn't contain anything.  So swap this and item and
                // then destruct item.
                item.apply_to_contents(*this);
                item.destruct();
            }
            else if (item.type_identity == 0)
            {
                // *this doesn't contain anything.  So swap this and item and
                // then destruct this.
                apply_to_contents(item);
                destruct();
            }
            else
            {
                type_safe_union temp;
                // swap *this into temp
                apply_to_contents(temp);
                // swap item into *this
                item.apply_to_contents(*this);
                // swap temp into item
                temp.apply_to_contents(item);
            }
        }

        template <typename T> T& get() { construct<T>();  return *reinterpret_cast<T*>(mem.data); }

    };

// ----------------------------------------------------------------------------------------

    template <
        typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10
        >
    inline void swap (
        type_safe_union<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>& a, 
        type_safe_union<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>& b 
    ) { a.swap(b); }   

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_TYPE_SAFE_UNIOn_h_

