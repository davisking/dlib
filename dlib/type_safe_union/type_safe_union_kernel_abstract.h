// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_TYPE_SAFE_UNION_KERNEl_ABSTRACT_
#ifdef DLIB_TYPE_SAFE_UNION_KERNEl_ABSTRACT_

#include "../algs.h"
#include "../noncopyable.h"

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
            REQUIREMENTS ON ALL TEMPLATE ARGUMENTS
                All template arguments must be default constructable and have
                a global swap.

            INITIAL VALUE
                - is_empty() == true
                - contains<U>() == false, for all possible values of U

            WHAT THIS OBJECT REPRESENTS 
                This object is a type safe analogue of the classic C union object. 
                The type_safe_union, unlike a union, can contain non-POD types such 
                as std::string.  

                For example:
                    union my_union
                    {
                        int a;
                        std::string b;   // Error, std::string isn't a POD
                    };

                    type_safe_union<int,std::string> my_type_safe_union;  // No error
        !*/

    public:

        type_safe_union(
        );
        /*!
            ensures
                - this object is properly initialized
        !*/

        ~type_safe_union(
        );
        /*!
            ensures
                - all resources associated with this object have been freed
        !*/

        template <typename T>
        bool contains (
        ) const;
        /*!
            ensures
                - if (this type_safe_union currently contains an object of type T) then
                    - returns true
                - else
                    - returns false
        !*/

        bool is_empty (
        ) const;
        /*!
            ensures
                - if (this type_safe_union currently contains any object at all) then
                    - returns true
                - else
                    - returns false
        !*/

        template <typename T>
        void apply_to_contents (
            T& obj
        );
        /*!
            requires
                - obj is a function object capable of operating on all the types contained
                  in this type_safe_union.  I.e.  obj(this->get<U>()) must be a valid
                  expression for all the possible U types.
            ensures
                - if (is_empty() == false) then
                    - Let U denote the type of object currently contained in this type_safe_union
                    - calls obj(this->get<U>())
        !*/

        template <typename T>
        void apply_to_contents (
            const T& obj
        );
        /*!
            requires
                - obj is a function object capable of operating on all the types contained
                  in this type_safe_union.  I.e.  obj(this->get<U>()) must be a valid
                  expression for all the possible U types.
            ensures
                - if (is_empty() == false) then
                    - Let U denote the type of object currently contained in this type_safe_union
                    - calls obj(this->get<U>())
        !*/

        template <typename T> 
        T& get(
        );
        /*!
            ensures
                - #is_empty() == false
                - #contains<T>() == true
                - if (contains<T>() == true)
                    - returns a non-const reference to the object contained in this type_safe_union.
                - else
                    - Constructs an object of type T inside *this
                    - Any previous object stored in this type_safe_union is destructed and its
                      state is lost.
                    - returns a non-const reference to the newly created T object.

        !*/

        void swap (
            type_safe_union& item
        );
        /*!
            ensures
                - swaps *this and item
        !*/

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
    /*!
        provides a global swap function
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10
        >
    void serialize (
        const type_safe_union<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>& item, 
        std::istream& in
    );   
    /*!
        provides serialization support 
    !*/

    template <
        typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename T8, typename T9, typename T10
        >
    void deserialize (
        type_safe_union<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>& item, 
        std::istream& in
    );   
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_TYPE_SAFE_UNION_KERNEl_ABSTRACT_

