// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_TYPE_SAFE_UNION_KERNEl_ABSTRACT_
#ifdef DLIB_TYPE_SAFE_UNION_KERNEl_ABSTRACT_

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class bad_type_safe_union_cast : public std::bad_cast 
    {
        /*!
            This is the exception object thrown by type_safe_union::cast_to() if the
            type_safe_union does not contain the type of object being requested.
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <typename... Types>
    class type_safe_union
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
        ) = default;
        /*!
            ensures
                - this object is properly initialized
        !*/

        type_safe_union (
            const type_safe_union& item
        )
        /*!
            ensures
                - copy constructs *this from item
        !*/

        template <
            typename T,
            typename = typename std::enable_if<is_valid<T>::value>::type
        >
        type_safe_union (
            const T& item
        );
        /*!
            ensures
                - constructs *this from item
                - #get<T>() == item
                  (i.e. this object will contain a copy of item)
        !*/

        type_safe_union& operator=(
            const type_safe_union& item
        );
        /*!
            ensures
                - copy assigns *this from item
        !*/

        template <
            typename T,
            typename = typename std::enable_if<is_valid<T>::value>::type
        >
        type_safe_union& operator= (
            const T& item
        );
        /*!
            ensures
                - copy assigns *this from item
                - #get<T> == item
                  (i.e. this object will contain a copy of item)
        !*/

        type_safe_union (
            type_safe_union&& item
        );
        /*!
            ensures
                - move constructs *this from item.
        !*/

        type_safe_union& operator= (
            type_safe_union&& item
        );
        /*!
            ensures
                - move assigns *this from item.
        !*/

        template <
            typename T,
            typename = typename std::enable_if<is_valid<T>::value>::type
        >
        type_safe_union (
            T&& item
        );
        /*!
            ensures
                - move constructs *this from item
                - #get<T> == item
                  (i.e. this object will have moved item into *this)
        !*/

         template <
            typename T,
            typename = typename std::enable_if<is_valid<T>::value>::type
        >
        type_safe_union& operator= (
            T&& item
        );
        /*!
            ensures
                - move assigns *this from item
                - #get<T> == item
                  (i.e. this object will have moved item into *this)
        !*/

        template <
            typename T,
            typename... Args
        >
        type_safe_union (
            in_place_tag<T>,
            Args&&... args
        );
        /*!
            ensures
                - constructs *this with type T using constructor-arguments args...
                  (i.e. this object will have moved item into *this)
        !*/

        ~type_safe_union(
        );
        /*!
            ensures
                - all resources associated with this object have been freed
        !*/

        void clear();
        /*!
            ensures
                - all resources associated with this object have been freed
                - #is_empty() == true
        !*/

        template <
            typename T,
            typename... Args
        >
        void emplace(
            Args&&... args
        );
        /*!
            ensures
                - re-assigns *this with type T using constructor-arguments args...
                  (i.e. this object will have moved item into *this)
        !*/

        template <typename T>
        static int get_type_id (
        );
        /*!
           ensures
              - if (T is the same type as one of the template arguments) then
                 - returns a number indicating which template argument it is.
                   (e.g. if T is the same type as T3 then this function returns 3)
               - else
                  - returns -1
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

        int index(
        ) const;
        /*!
            ensures
                - returns type_identity, i.e, the index of the currently held type.
                  For example if the current type is the first type defined in the template parameters then
                    - returns 1
                  If the current type is the second type in the template parameters then
                    - returns 2
                  etc.
                  If the current object is empty, i.e. is_empty() == true, then
                    - returns 0
        !*/

        template <typename F>
        void apply_to_contents(
            F&& f
        );
        /*!
            requires
                - f is a callable object capable of operating on all the types contained
                  in this type_safe_union.  I.e.  std::forward<F>(f)(this->get<U>()) must be a valid
                  expression for all the possible U types.
            ensures
                - if (is_empty() == false) then
                    - Let U denote the type of object currently contained in this type_safe_union
                    - calls std::forward<F>(f)(this->get<U>())
                    - The object returned by this->get<U>() will be non-const
        !*/

        template <typename F>
        void apply_to_contents(
            F&& f
        ) const;
        /*!
            requires
                - f is a callable object capable of operating on all the types contained
                  in this type_safe_union.  I.e.  std::forward<F>(f)(this->get<U>()) must be a valid
                  expression for all the possible U types.
            ensures
                - if (is_empty() == false) then
                    - Let U denote the type of object currently contained in this type_safe_union
                    - calls std::forward<F>(f)(this->get<U>())
                    - The object returned by this->get<U>() will be non-const
        !*/

        template <typename F>
        auto visit(
            F&& f
        );
        /*!
            requires
                - f is a callable object capable of operating on all the types contained
                  in this type_safe_union.  I.e.  std::forward<F>(f)(this->get<U>()) must be a valid
                  expression for all the possible U types.
            ensures
                - if (is_empty() == false) then
                    - Let U denote the type of object currently contained in this type_safe_union
                    - calls std::forward<F>(f)(this->get<U>())
                    - The object returned by this->get<U>() will be non-const
        !*/

        template <typename F>
        auto visit(
            F&& f
        );
        /*!
            requires
                - f is a callable object capable of operating on all the types contained
                  in this type_safe_union.  I.e.  std::forward<F>(f)(this->get<U>()) must be a valid
                  expression for all the possible U types.
            ensures
                - if (is_empty() == false) then
                    - Let U denote the type of object currently contained in this type_safe_union
                    - calls std::forward<F>(f)(this->get<U>())
                    - The object returned by this->get<U>() will be non-const
        !*/

        template <typename T> 
        T& get(
        );
        /*!
            requires
                - T must be one of the types given to this object's template arguments
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

        template <typename T>
        const T& cast_to (
        ) const;
        /*!
            requires
                - T must be one of the types given to this object's template arguments
            ensures
                - if (contains<T>() == true) then
                    - returns a const reference to the object contained in this type_safe_union.
                - else
                    - throws bad_type_safe_union_cast
        !*/

        template <typename T>
        T& cast_to (
        );
        /*!
            requires
                - T must be one of the types given to this object's template arguments
            ensures
                - if (contains<T>() == true) then
                    - returns a non-const reference to the object contained in this type_safe_union.
                - else
                    - throws bad_type_safe_union_cast
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

    template < ...  >
    inline void swap (
        type_safe_union<...>& a, 
        type_safe_union<...>& b 
    ) { a.swap(b); }   
    /*!
        provides a global swap function
    !*/

// ----------------------------------------------------------------------------------------

    template < ... >
    void serialize (
        const type_safe_union<...>& item, 
        std::ostream& out 
    );   
    /*!
        provides serialization support 

        Note that type_safe_union objects are serialized as follows:
         - if (item.is_empty()) then
            - perform: serialize(0, out)
         - else
            - perform: serialize(item.get_type_id<type_of_object_in_item>(), out);
                       serialize(item.get<type_of_object_in_item>(), out);
    !*/

    template < ...  >
    void deserialize (
        type_safe_union<...>& item, 
        std::istream& in
    );   
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_TYPE_SAFE_UNION_KERNEl_ABSTRACT_

