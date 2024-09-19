// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_AnY_FUNCTION_ABSTRACT_H_
#ifdef DLIB_AnY_FUNCTION_ABSTRACT_H_

#include "any_abstract.h"
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        class Storage, 
        typename function_type
        >
    class any_function_basic
    {
        /*!
            REQUIREMENTS ON Storage
                This must be one of the storage types from dlib/any/storage.hh
                E.g. storage_heap, storage_stack, etc.

                It determines the method by which any_function_basic holds onto the function it uses.

            REQUIREMENTS ON function_type
                This type should be a function signature.  Some examples are:
                    void (int,int)  // a function returning nothing and taking two ints
                    void ()         // a function returning nothing and taking no arguments
                    char (double&)  // a function returning a char and taking a reference to a double

                The number of arguments in the function must be no greater than 10.

            INITIAL VALUE
                - is_empty() == true
                - for all T: contains<T>() == false

            WHAT THIS OBJECT REPRESENTS
                This object is a version of dlib::any that is restricted to containing 
                elements which are some kind of function object with an operator() which
                matches the function signature defined by function_type.


                Here is an example:
                    #include <iostream>
                    #include <string>
                    #include "dlib/any.h"
                    void print_message(string str) { cout << str << endl; }

                    int main()
                    {
                        dlib::any_function<void(string)> f;
                        f = print_message;
                        f("hello world"); // calls print_message("hello world")
                    }

                Note that any_function_basic objects can be used to store general function 
                objects (i.e. defined by a class with an overloaded operator()) in
                addition to regular global functions.  
        !*/

    public:

        // This is the type of object returned by function_type functions.
        typedef result_type_for_function_type result_type;

        any_function_basic(
        );
        /*!
            ensures
                - this object is properly initialized
        !*/

        any_function_basic (
            const any_function_basic& item
        );
        /*!
            ensures
                - copies the state of item into *this.  
                - Note that *this and item will contain independent copies of the
                  contents of item.  That is, this function performs a deep
                  copy and therefore does not result in *this containing
                  any kind of reference to item.
        !*/

        any_function_basic (
            any_function_basic&& item
        );
        /*!
            ensures
                - moves item into *this.
                - The exact move semantics are determined by which Storage type is used.  E.g. 
                  storage_heap will result in #item.is_empty()==true but storage_view would result
                  in #item.is_empty() == false
        !*/

        template < typename Funct >
        any_function_basic (
            Funct&& funct 
        );
        /*!
            ensures
                - #contains<T>() == true
                - #cast_to<T>() == item
                  (i.e. calling operator() will invoke funct())
        !*/

        void clear (
        );
        /*!
            ensures
                - #*this will have its default value.  I.e. #is_empty() == true
        !*/

        template <typename T>
        bool contains (
        ) const;
        /*!
            ensures
                - if (this object currently contains an object of type T) then
                    - returns true
                - else
                    - returns false
        !*/

        bool is_empty(
        ) const;
        /*!
            ensures
                - if (this object contains any kind of object) then
                    - returns false 
                - else
                    - returns true 
        !*/

        bool is_set (
        ) const;
        /*!
            ensures
                - returns !is_empty()
        !*/

        explicit operator bool(
        ) const;
        /*!
            ensures
                - returns is_set()
        !*/

        result_type operator(Args... args) (
        ) const;
        /*!
            requires
                - is_empty() == false
                - the signature defined by function_type takes no arguments
            ensures
                - Let F denote the function object contained within *this.  Then
                  this function performs:
                    return F(std::forward<Args>(args)...)
        !*/

        template <typename T>
        T& cast_to(
        );
        /*!
            ensures
                - if (contains<T>() == true) then
                    - returns a non-const reference to the object contained within *this
                - else
                    - throws bad_any_cast
        !*/

        template <typename T>
        const T& cast_to(
        ) const;
        /*!
            ensures
                - if (contains<T>() == true) then
                    - returns a const reference to the object contained within *this
                - else
                    - throws bad_any_cast
        !*/

        template <typename T>
        T& get(
        );
        /*!
            ensures
                - #is_empty() == false
                - #contains<T>() == true
                - if (contains<T>() == true)
                    - returns a non-const reference to the object contained in *this.
                - else
                    - Constructs an object of type T inside *this
                    - Any previous object stored in this any_function_basic object is destructed and its
                      state is lost.
                    - returns a non-const reference to the newly created T object.
        !*/

        any_function_basic& operator= (
            const any_function_basic& item
        );
        /*!
            ensures
                - copies the state of item into *this.  
                - Note that the type of copy is determined by the Storage template argument.  E.g.
                  storage_sbo will result in a deep copy, while storage_view would result in *this
                  and item referring to the same underlying function.
        !*/

        void swap (
            any_function_basic& item
        );
        /*!
            ensures
                - swaps *this and item
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        typename function_type
        > 
    T& any_cast(
        any_function_basic<function_type>& a
    ) { return a.cast_to<T>(); }
    /*!
        ensures
            - returns a.cast_to<T>()
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        typename function_type
        > 
    const T& any_cast(
        const any_function_basic<function_type>& a
    ) { return a.cast_to<T>(); }
    /*!
        ensures
            - returns a.cast_to<T>()
    !*/

// ----------------------------------------------------------------------------------------

    /*!A any_function

        A version of any_function_basic (defined above) that owns the function it contains.  Uses
        the small buffer optimization to make working with small lambdas faster.
    !*/
    template <class F> 
    using any_function = any_function_basic<te::storage_sbo<16>, F>;

    /*!A any_function_view

        A version of any_function_basic (defined above) that *DOES NOT* own the function it
        contains.  It merely holds a pointer to the function given to its constructor. 
    !*/
    template <class F> 
    using any_function_view = any_function_basic<te::storage_view, F>;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_AnY_FUNCTION_ABSTRACT_H_

