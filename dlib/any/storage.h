// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TYPE_ERASURE_H_
#define DLIB_TYPE_ERASURE_H_

#include <type_traits>
#include <utility>
#include <typeindex>
#include <new>
#include <memory>
#include "../assert.h"

namespace dlib
{

// -----------------------------------------------------------------------------------------------------

    class bad_any_cast : public std::bad_cast 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is the exception class used by the storage objects.
                It is used to indicate when someone attempts to cast a storage
                object into a type which isn't contained in the object.
        !*/
        
    public:
          virtual const char * what() const throw()
          {
              return "bad_any_cast";
          }
    };
    
// -----------------------------------------------------------------------------------------------------

    namespace te
    {

        /*!
            This is used as a SFINAE tool to prevent a function taking a universal reference from
            binding to some undesired type.  For example:
                template <
                    typename T,
                    T_is_not_this_type<SomeExcludedType, T> = true
                    >
                void foo(T&&);
            prevents foo() from binding to an object of type SomeExcludedType.
        !*/
        template<class Storage, class T>
        using T_is_not_this_type = std::enable_if_t<!std::is_same<std::decay_t<T>, Storage>::value, bool>;

// -----------------------------------------------------------------------------------------------------

        template<class Storage>
        class storage_base
        {       
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class defines functionality common to all type erasure storage objects
                    (defined below in this file).  These objects are essentially type-safe versions of
                    a void*.  In particular, they are containers which can contain only one object
                    but the object may be of any type.  

                    Each storage object implements a different way of storing the underlying object.
                    E.g. on the heap or stack or some other more specialized method.
            !*/

        public:

            bool is_empty() const
            /*!
                ensures
                    - if (this object contains any kind of object) then
                        - returns false 
                    - else
                        - returns true
            !*/
            {
                const Storage& me = *static_cast<const Storage*>(this);
                return me.get_ptr() == nullptr;
            }

            template<typename T>
            bool contains() const
            /*!
                ensures
                    - if (this object currently contains an object of type T) then
                        - returns true
                    - else
                        - returns false
            !*/
            {
                const Storage& me = *static_cast<const Storage*>(this);
                return !is_empty() && me.type_id() == std::type_index{typeid(T)};
            }

            template<typename T>
            T& unsafe_get() 
            /*!
                requires
                    - contains<T>() == true
                ensures
                    - returns a reference to the object contained within *this.
            !*/
            {
                DLIB_ASSERT(contains<T>());
                Storage& me = *static_cast<Storage*>(this);
                return *reinterpret_cast<T*>(me.get_ptr()); 
            }

            template<typename T>
            const T& unsafe_get() const 
            /*!
                requires
                    - contains<T>() == true
                ensures
                    - returns a const reference to the object contained within *this.
            !*/
            {
                DLIB_ASSERT(contains<T>());
                const Storage& me = *static_cast<const Storage*>(this);
                return *reinterpret_cast<const T*>(me.get_ptr()); 
            }
            
            template <typename T>
            T& get(
            ) 
            /*!
                ensures
                    - #is_empty() == false
                    - #contains<T>() == true
                    - if (contains<T>() == true)
                        - returns a non-const reference to the object contained in *this.
                    - else
                        - Constructs an object of type T inside *this
                        - Any previous object stored in this any object is destructed and its
                          state is lost.
                        - returns a non-const reference to the newly created T object.
            !*/
            {
                Storage& me = *static_cast<Storage*>(this);

                if (!contains<T>())
                    me = T{};
                return unsafe_get<T>();
            }

            template <typename T>
            T& cast_to(
            ) 
            /*!
                ensures
                    - if (contains<T>() == true) then
                        - returns a non-const reference to the object contained within *this
                    - else
                        - throws bad_any_cast
            !*/
            {
                if (!contains<T>())
                    throw bad_any_cast{};
                return unsafe_get<T>();
            }

            template <typename T>
            const T& cast_to(
            ) const
            /*!
                ensures
                    - if (contains<T>() == true) then
                        - returns a const reference to the object contained within *this
                    - else
                        - throws bad_any_cast
            !*/
            {
                if (!contains<T>())
                    throw bad_any_cast{};
                return unsafe_get<T>();
            }
        };

// -----------------------------------------------------------------------------------------------------

        class storage_heap : public storage_base<storage_heap>
        {
        public:
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.

                    This particular storage type uses heap allocation only.
            !*/

            storage_heap() = default;
            /*!
                ensures
                    - #is_empty() == true
                    - for all T: #contains<T>() == false
            !*/

            template <
                class T,
                class T_ = std::decay_t<T>,
                T_is_not_this_type<storage_heap, T> = true
            >
            storage_heap(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - copies or moves the incoming object (depending on the forwarding reference)
                    - #is_empty() == false 
                    - #contains<std::decay_t<T>>() == true
                    - #unsafe_get<T>() will yield the provided t.
            !*/
            :   ptr{new T_{std::forward<T>(t)}},
                del{[](void *self) { 
                    delete reinterpret_cast<T_*>(self); 
                }},
                copy{[](const void *self) -> void * {
                    return new T_{*reinterpret_cast<const T_*>(self)};
                }},
                type_id_{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            storage_heap(const storage_heap& other)
            /*!
                ensures
                    - #is_empty() == other.is_empty()
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor.
            !*/
            :   ptr{other.ptr ? other.copy(other.ptr) : nullptr},
                del{other.del},
                copy{other.copy},
                type_id_{other.type_id_}
            {
            }

            storage_heap& operator=(const storage_heap& other)
            /*!
                ensures
                    - if is_empty() == false then
                       - destructs the object contained in this class.
                    - #is_empty() == other.is_empty()
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor.
            !*/
            {
                if (this != &other) 
                    *this = std::move(storage_heap{other});
                return *this;
            }

            storage_heap(storage_heap&& other) noexcept
            /*!
                ensures
                    - The state of other is moved into *this.
                    - #other.is_empty() == true
            !*/
            :   ptr{std::exchange(other.ptr, nullptr)},
                del{std::exchange(other.del, nullptr)},
                copy{std::exchange(other.copy, nullptr)},
                type_id_{std::exchange(other.type_id_, nullptr)}
            {
            }

            storage_heap& operator=(storage_heap&& other) noexcept
            /*!
                ensures
                    - The state of other is moved into *this.
                    - #other.is_empty() == true
                    - returns *this
            !*/
            {
                if (this != &other) 
                {
                    clear();
                    ptr     = std::exchange(other.ptr, nullptr);
                    del     = std::exchange(other.del, nullptr);
                    copy    = std::exchange(other.copy, nullptr);
                    type_id_ = std::exchange(other.type_id_, nullptr);
                }
                return *this;
            }

            ~storage_heap()
            /*!
                ensures
                    - destructs the object contained in *this if one exists. 
            !*/
            {
                if (ptr)
                    del(ptr);
            }

            void clear()
            /*!
                ensures
                    - #is_empty() == true
            !*/
            {
                storage_heap{std::move(*this)};
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return ptr;
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return ptr;
            }

            std::type_index type_id() const
            /*!
                requires
                    - is_empty() == false
                ensures
                    - returns the std::type_index of the type contained within this object.
                      I.e. if this object contains the type T then this returns std::type_index{typeid(T)}.
            !*/
            {
                DLIB_ASSERT(!this->is_empty());
                return type_id_();
            }

        private:
            void* ptr                     = nullptr;
            void  (*del)(void*)           = nullptr;
            void* (*copy)(const void*)    = nullptr;
            std::type_index (*type_id_)()  = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

        template <std::size_t Size, std::size_t Alignment = 8>
        class storage_stack : public storage_base<storage_stack<Size, Alignment>>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.

                    This particular storage type uses stack allocation using a template size and alignment.
                    Therefore, only objects whose size and alignment fits the template parameters can be
                    erased and absorbed into this object.  Attempting to store a type not
                    representable on the stack with those settings will result in a build error.

                    This object will be capable of storing any type with an alignment requirement
                    that is a divisor of Alignment.
            !*/

        public:
            storage_stack() = default;
            /*!
                ensures
                    - #is_empty() == true
                    - for all T: #contains<T>() == false
            !*/

            template <
                class T,
                class T_ = std::decay_t<T>,
                T_is_not_this_type<storage_stack, T> = true
            >
            storage_stack(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - copies or moves the incoming object (depending on the forwarding reference)
                    - #is_empty() == false 
                    - #contains<std::decay_t<T>>() == true
            !*/
            :   del{[](storage_stack& self) {
                    reinterpret_cast<T_*>(&self.data)->~T_();
                    self.del        = nullptr;
                    self.copy       = nullptr;
                    self.move       = nullptr;
                    self.type_id_   = nullptr;
                }},
                copy{[](const storage_stack& src, storage_stack& dst) {
                    new (&dst.data) T_{*reinterpret_cast<const T_*>(&src.data)};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id_ = src.type_id_;
                }},
                move{[](storage_stack& src, storage_stack& dst) {
                    new (&dst.data) T_{std::move(*reinterpret_cast<T_*>(&src.data))};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id_ = src.type_id_;
                }},
                type_id_{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
                static_assert(sizeof(T_) <= Size, "insufficient size");
                static_assert(Alignment % alignof(T_) == 0, "bad alignment");
                new (&data) T_{std::forward<T>(t)};
            }

            storage_stack(const storage_stack& other)
            /*!
                ensures
                    - #is_empty() == other.is_empty()
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor.
            !*/
            {
                if (other.copy)
                    other.copy(other, *this);
            }

            storage_stack& operator=(const storage_stack& other)
            /*!
                ensures
                    - #is_empty() == other.is_empty()
                    - if is_empty() == false then
                       - destructs the object contained in this class.
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor
            !*/
            {
                if (this != &other) 
                {
                    clear();
                    if (other.copy)
                        other.copy(other, *this);
                }
                return *this;
            }

            storage_stack(storage_stack&& other)
            /*!
                ensures
                    - #is_empty() == other.is_empty()
                    - if other.is_empty() == false then
                        - underlying object of other is moved using erased type's moved constructor
            !*/
            {
                if (other.move)
                    other.move(other, *this);
            }

            storage_stack& operator=(storage_stack&& other)
            /*!
                ensures
                    - if is_empty() == false then
                       - destructs the object contained in this class.
                    - #is_empty() == other.is_empty()
                    - if other.is_empty() == false then
                        - underlying object of other is moved using erased type's moved constructor.
                          This does not make other empty.  It will still contain a moved from object
                          of the underlying type in whatever that object's moved from state is.
                        - #other.is_empty() == false
            !*/
            {
                if (this != &other) 
                {
                    clear();
                    if (other.move)
                        other.move(other, *this);
                }
                return *this;
            }

            ~storage_stack()
            /*!
                ensures
                    - destructs the object contained in *this if one exists. 
            !*/
            {
                clear();
            }

            void clear()
            /*!
                ensures
                    - #is_empty() == true
            !*/
            {
                if (del)
                    del(*this);
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return del ? (void*)&data : nullptr;
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return del ? (const void*)&data : nullptr;
            }

            std::type_index type_id() const
            /*!
                requires
                    - is_empty() == false
                ensures
                    - returns the std::type_index of the type contained within this object.
                      I.e. if this object contains the type T then this returns std::type_index{typeid(T)}.
            !*/
            {
                DLIB_ASSERT(!this->is_empty());
                return type_id_();
            }

        private:
            std::aligned_storage_t<Size, Alignment> data;
            void (*del)(storage_stack&)                         = nullptr;
            void (*copy)(const storage_stack&, storage_stack&)  = nullptr;
            void (*move)(storage_stack&, storage_stack&)        = nullptr;
            std::type_index (*type_id_)()                        = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

        template <std::size_t Size, std::size_t Alignment = 8>
        class storage_sbo : public storage_base<storage_sbo<Size, Alignment>>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.

                    This particular storage type uses small buffer optimization (SBO), i.e. optional
                    stack allocation if the erased type has sizeof <= Size and alignment
                    requirements no greater than the given Alignment template value.  If not it
                    allocates the object on the heap.
            !*/

        public:
            // type_fits<T>::value tells us if our SBO can hold T.
            template<typename T>
            struct type_fits : std::integral_constant<bool, sizeof(T) <= Size && Alignment % alignof(T) == 0>{};

            storage_sbo() = default;
            /*!
                ensures
                    - #is_empty() == true
                    - for all T: #contains<T>() == false
            !*/

            template <
                class T,
                class T_ = std::decay_t<T>,
                T_is_not_this_type<storage_sbo, T> = true,
                std::enable_if_t<type_fits<T_>::value, bool> = true
            >
            storage_sbo(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - copies or moves the incoming object (depending on the forwarding reference)
                    - #is_empty() == false 
                    - #contains<std::decay_t<T>>() == true
                    - stack allocation is used
            !*/
            :   ptr{new (&data) T_{std::forward<T>(t)}},
                del{[](storage_sbo& self) {
                    reinterpret_cast<T_*>(&self.data)->~T_();
                    self.ptr        = nullptr;
                    self.del        = nullptr;
                    self.copy       = nullptr;
                    self.move       = nullptr;
                    self.type_id_    = nullptr;
                }},
                copy{[](const storage_sbo& src, storage_sbo& dst) {
                    dst.ptr     = new (&dst.data) T_{*reinterpret_cast<const T_*>(src.ptr)};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id_ = src.type_id_;
                }},
                move{[](storage_sbo& src, storage_sbo& dst) {
                    dst.ptr     = new (&dst.data) T_{std::move(*reinterpret_cast<T_*>(src.ptr))};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id_ = src.type_id_;
                }},
                type_id_{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            template <
                class T,
                class T_ = std::decay_t<T>,
                T_is_not_this_type<storage_sbo, T> = true,
                std::enable_if_t<!type_fits<T_>::value, bool> = true
            >
            storage_sbo(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - copies or moves the incoming object (depending on the forwarding reference)
                    - #is_empty() == false
                    - #contains<std::decay_t<T>>() == true
                    - heap allocation is used
            !*/
            :   ptr{new T_{std::forward<T>(t)}},
                del{[](storage_sbo& self) {
                    delete reinterpret_cast<T_*>(self.ptr);
                    self.ptr        = nullptr;
                    self.del        = nullptr;
                    self.copy       = nullptr;
                    self.move       = nullptr;
                    self.type_id_    = nullptr;
                }},
                copy{[](const storage_sbo& src, storage_sbo& dst) {
                    dst.ptr     = new T_{*reinterpret_cast<const T_*>(src.ptr)};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id_ = src.type_id_;
                }},
                move{[](storage_sbo& src, storage_sbo& dst) {
                    dst.ptr     = std::exchange(src.ptr,     nullptr);
                    dst.del     = std::exchange(src.del,     nullptr);
                    dst.copy    = std::exchange(src.copy,    nullptr);
                    dst.move    = std::exchange(src.move,    nullptr);
                    dst.type_id_ = std::exchange(src.type_id_, nullptr);
                }},
                type_id_{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            storage_sbo(const storage_sbo& other)
            /*!
                ensures
                    - #is_empty() == other.is_empty()
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor
            !*/
            {
                if (other.copy)
                    other.copy(other, *this);
            }

            storage_sbo& operator=(const storage_sbo& other)
            /*!
                ensures
                    - if is_empty() == false then
                       - destructs the object contained in this class.
                    - #is_empty() == other.is_empty()
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor
            !*/
            {
                if (this != &other) 
                {
                    clear();
                    if (other.copy)
                        other.copy(other, *this);
                }
                return *this;
            }

            storage_sbo(storage_sbo&& other)
            /*!
                ensures
                    - #is_empty() == other.is_empty()
                    - if other.is_empty() == false then
                        - if underlying object of other is allocated on stack then
                            - underlying object of other is moved using erased type's moved constructor.
                              This does not make other empty.  It will still contain a moved from
                              object of the underlying type in whatever that object's moved from
                              state is.
                            - #other.is_empty() == false
                        - else
                            - storage heap pointer is moved.
                            - #other.is_empty() == true 
            !*/
            {
                if (other.move)
                    other.move(other, *this);
            }

            storage_sbo& operator=(storage_sbo&& other)
            /*!
                ensures
                    - underlying object is destructed if is_empty() == false
                    - #is_empty() == other.is_empty()
                    - if other.is_empty() == false then
                        - if underlying object of other is allocated on stack then
                            - underlying object of other is moved using erased type's moved constructor.
                              This does not make other empty.  It will still contain a moved from
                              object of the underlying type in whatever that object's moved from
                              state is.
                            - #other.is_empty() == false
                        - else
                            - storage heap pointer is moved.
                            - #other.is_empty() == true 
            !*/
            {
                if (this != &other) 
                {
                    clear();
                    if (other.move)
                        other.move(other, *this);
                }
                return *this;
            }

            ~storage_sbo()
            /*!
                ensures
                    - destructs the object contained in *this if one exists. 
            !*/
            {
                clear();
            }

            void clear()
            /*!
                ensures
                    - #is_empty() == true
            !*/
            {
                if (ptr)
                    del(*this);
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return ptr;
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return ptr;
            }

            std::type_index type_id() const
            /*!
                requires
                    - is_empty() == false
                ensures
                    - returns the std::type_index of the type contained within this object.
                      I.e. if this object contains the type T then this returns std::type_index{typeid(T)}.
            !*/
            {
                DLIB_ASSERT(!this->is_empty());
                return type_id_();
            }

        private:
            std::aligned_storage_t<Size, Alignment> data;
            void* ptr                                       = nullptr;
            void (*del)(storage_sbo&)                       = nullptr;
            void (*copy)(const storage_sbo&, storage_sbo&)  = nullptr;
            void (*move)(storage_sbo&, storage_sbo&)        = nullptr;
            std::type_index (*type_id_)()                    = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

        class storage_shared : public storage_base<storage_shared>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.

                    This particular storage type uses std::shared_ptr<void> to store and erase
                    incoming objects. Therefore, it uses heap allocation and reference counting.
                    Moreover, it has the same copying and move semantics as std::shared_ptr.  I.e.
                    it results in the underlying object being held by reference rather than by
                    value.
            !*/

        public:
            storage_shared() = default;
            /*!
                ensures
                    - #is_empty() == true
                    - for all T: #contains<T>() == false
            !*/

            template <
                class T,
                class T_ = std::decay_t<T>,
                T_is_not_this_type<storage_shared, T> = true
            >
            storage_shared(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - copies or moves the incoming object (depending on the forwarding reference)
                    - #is_empty() == true
                    - #contains<std::decay_t<T>>() == true
            !*/
            :   ptr{std::make_shared<T_>(std::forward<T>(t))},
                type_id_{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            // This object has the same copy/move semantics as a std::shared_ptr<void>
            storage_shared(const storage_shared& other)             = default;
            storage_shared& operator=(const storage_shared& other)  = default;
            storage_shared(storage_shared&& other) noexcept = default;
            storage_shared& operator=(storage_shared&& other) noexcept = default;

            void clear() 
            /*!
                ensures
                    - #is_empty() == true
            !*/
            { 
                ptr     = nullptr;
                type_id_ = nullptr;
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return ptr.get();
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return ptr.get();
            }

            std::type_index type_id() const
            /*!
                requires
                    - is_empty() == false
                ensures
                    - returns the std::type_index of the type contained within this object.
                      I.e. if this object contains the type T then this returns std::type_index{typeid(T)}.
            !*/
            {
                DLIB_ASSERT(!this->is_empty());
                return type_id_();
            }

        private:
            std::shared_ptr<void> ptr    = nullptr;
            std::type_index (*type_id_)() = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

        class storage_view : public storage_base<storage_view>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.

                    This particular storage type is a view type, similar to std::string_view or
                    std::span.  So underlying objects are only ever referenced, not copied, moved or
                    destructed.  That is, instances of this object take no ownership of the objects
                    they contain.  So they are only valid as long as the contained object exists.
                    So storage_view merely holds a pointer to the underlying object.
            !*/

        public:
            storage_view() = default;
            /*!
                ensures
                    - #is_empty() == true
                    - for all T: #contains<T>() == false
            !*/
            
            template <
                class T,
                class T_ = std::decay_t<T>,
                T_is_not_this_type<storage_view, T> = true
            >
            storage_view(T &&t) noexcept
            /*!
                ensures
                    - #get_ptr() == &t
                    - #is_empty() == false 
                    - #contains<std::decay_t<T>>() == true
            !*/
            :   ptr{&t},
                type_id_{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }
            
            // This object has the same copy/move semantics as a void*.
            storage_view(const storage_view& other)             = default;
            storage_view& operator=(const storage_view& other)  = default;
            storage_view(storage_view&& other) noexcept = default;
            storage_view& operator=(storage_view&& other) noexcept = default;

            void clear() 
            /*!
                ensures
                    - #is_empty() == true
            !*/
            { 
                ptr     = nullptr;
                type_id_ = nullptr;
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return ptr;
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object or nullptr if is_empty()
            !*/
            {
                return ptr;
            }

            std::type_index type_id() const
            /*!
                requires
                    - is_empty() == false
                ensures
                    - returns the std::type_index of the type contained within this object.
                      I.e. if this object contains the type T then this returns std::type_index{typeid(T)}.
            !*/
            {
                DLIB_ASSERT(!this->is_empty());
                return type_id_();
            }

        private:
            void* ptr = nullptr;
            std::type_index (*type_id_)() = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_TYPE_ERASURE_H_
