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

    template<typename T>
    struct in_place_tag {};
    /*!
        This is an empty class type used as a special disambiguation tag to be
        passed as the first argument to the constructor of type_safe_union that performs
        in-place construction of an object.

        Here is an example of its usage:

        struct A
        {
            int i = 0;
            int j = 0;

            A(int i_, int j_) : i(i_), j(j_) {}
        };

        using tsu = type_safe_union<A,std::string>;

        tsu a(in_place_tag<A>{}, 0, 1);
    !*/

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
                - #is_empty() == true
        !*/

        type_safe_union (
            const type_safe_union& item
        )
        /*!
            ensures
                - copy constructs *this from item
        !*/

        type_safe_union& operator=(
            const type_safe_union& item
        );
        /*!
            ensures
                - copy assigns *this from item
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
            typename T
        >
        type_safe_union (
            T&& item
        );
        /*!
            requires
                - std::decay_t<T> must be one of the types given to this object's template arguments
            ensures
                - constructs *this from item using perfect forwarding (converting constructor)
                - #get<T>() == std::forward<T>(item)
                  (i.e. this object will either contain a copy of item or will have moved item into *this
                   depending on the reference type)
        !*/

        template <
            typename T
        >
        type_safe_union& operator= (
            T&& item
        );
        /*!
            requires
                - std::decay_t<T> must be one of the types given to this object's template arguments
            ensures
                - assigns *this from item using perfect forwarding (converting assignment)
                - #get<T> == std::forward<T>(item)
                  (i.e. this object will either contain a copy of item or will have moved item into *this
                   depending on the reference type)
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
            requires
                - T must be one of the types given to this object's template arguments
            ensures
                - constructs *this with type T using constructor-arguments args...
                  (i.e. efficiently performs *this = T(args...))
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
            requires
                - T must be one of the types given to this object's template arguments
            ensures
                - re-assigns *this with type T using constructor-arguments args...
                  (i.e. efficiently performs *this = T(args...))
        !*/

        template <typename T>
        static constexpr int get_type_id (
        );
        /*!
           ensures
              - if (T is the same type as one of the template arguments) then
                 - returns a number indicating which template argument it is. In particular,
                   if it's the first template argument it returns 1, if the second then 2, and so on.
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

        int get_current_type_id(
        ) const;
        /*!
            ensures
                - returns type_identity, i.e, the index of the currently held type.
                  For example if the current type is the first template argument it returns 1, if it's the second then 2, and so on.
                  If the current object is empty, i.e. is_empty() == true, then
                    - returns 0
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
                    - returns std::forward<F>(f)(this->get<U>())
                    - The object passed to f() (i.e. by this->get<U>()) will be non-const.
        !*/

        template <typename F>
        auto visit(
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
                    - returns std::forward<F>(f)(this->get<U>())
                    - The object passed to f() (i.e. by this->get<U>()) will be const.
        !*/

        template <typename F>
        void apply_to_contents(
            F&& f
        );
        /*!
            ensures:
                equivalent to calling visit(std::forward<F>(f)) with void return type
        !*/

        template <typename F>
        void apply_to_contents(
            F&& f
        ) const;
        /*!
            ensures:
                equivalent to calling visit(std::forward<F>(f)) with void return type
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

    template<typename... Types>
    inline void swap (
        type_safe_union<Types...>& a, 
        type_safe_union<Types...>& b 
    ) { a.swap(b); }   
    /*!
        provides a global swap function
    !*/

// ----------------------------------------------------------------------------------------

    template<typename... Types>
    void serialize (
        const type_safe_union<Types...>& item, 
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

    template<typename... Types>
    void deserialize (
        type_safe_union<Types...>& item, 
        std::istream& in
    );   
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------

    template<typename... T>
    overloaded_helper<typename std::decay<T>::type...> overloaded(T&&... t)
    {
        return overloaded_helper<typename std::decay<T>::type...>{std::forward<T>(t)...};
    }
    /*!
        This is a helper function for passing many callable objects (usually lambdas)
        to either apply_to_contents() or visit(), that combine to make a complete
        visitor. A picture paints a thousand words:

        using tsu = type_safe_union<int,float,std::string>;

        tsu a = std::string("hello there");

        std::string result;

        a.apply_to_contents(overloaded(
            [&result](int) {
                result = std::string("int");
            },
            [&result](float) {
                result = std::string("float");
            },
            [&result](const std::string& item) {
                result = item;
            }
        ));

        assert(result == "hello there");
        result = "";

        result = a.visit(overloaded(
            [](int) {
                return std::string("int");
            },
            [](float) {
                return std::string("float");
            },
            [](const std::string& item) {
                return item;
            }
        ));

        assert(result == "hello there");
    !*/
}

#endif // DLIB_TYPE_SAFE_UNION_KERNEl_ABSTRACT_
