// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TYPE_ERASURE_H_
#define DLIB_TYPE_ERASURE_H_

#include <type_traits>
#include <utility>
#include <typeindex>
#include <new>
#include <memory>

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

// -----------------------------------------------------------------------------------------------------

        template<class Storage, class T>
        using is_valid = std::enable_if_t<!std::is_same<std::decay_t<T>, Storage>::value, bool>;

// -----------------------------------------------------------------------------------------------------

        template<class Storage>
        struct storage_base
        {       
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
                return !is_empty() && me.type_id != nullptr && me.type_id() == std::type_index{typeid(T)};
            }

            template<typename T>
            T& unsafe_get() 
            /*!
                ensures
                    - returns a reference to the object contained within *this
                    - if *this doesn't contain an object of type T or any object at all, then it's undefined behaviour (UB)
            !*/
            {
                Storage& me = *static_cast<Storage*>(this);
                return *reinterpret_cast<T*>(me.get_ptr()); 
            }

            template<typename T>
            const T& unsafe_get() const 
            /*!
                ensures
                    - returns a const reference to the object contained within *this
                    - if *this doesn't contain an object of type T or any object at all, then it's undefined behaviour (UB)
            !*/
            {
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

        struct storage_heap : storage_base<storage_heap>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.
                    This particular storage type uses heap allocation only.
            !*/

            storage_heap() = default;
            /*!
                ensures
                    - this object is properly initialized
                    - is_empty() == true
                    - for all T: contains<T>() == false
            !*/

            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_heap, T> = true
            >
            storage_heap(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - Conversion constructor that copies or moves the incoming object (depending on the forwarding reference)
                    - is_empty() == true
                    - contains<std::decay_t<T>>() == true, otherwise contains<U>() == false for all other types
            !*/
            :   ptr{new T_{std::forward<T>(t)}},
                del{[](void *self) { 
                    delete reinterpret_cast<T_*>(self); 
                }},
                copy{[](const void *self) -> void * {
                    return new T_{*reinterpret_cast<const T_*>(self)};
                }},
                type_id{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            storage_heap(const storage_heap& other)
            /*!
                ensures
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor
                        - is_empty() == false
            !*/
            :   ptr{other.ptr ? other.copy(other.ptr) : nullptr},
                del{other.del},
                copy{other.copy},
                type_id{other.type_id}
            {
            }

            storage_heap& operator=(const storage_heap& other)
            /*!
                ensures
                    - underlying object is destructed if is_empty() == false
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor
                        - is_empty() == false
                    - else
                        - is_empty() == true
            !*/
            {
                if (this != &other) 
                    *this = std::move(storage_heap{other});
                return *this;
            }

            storage_heap(storage_heap&& other) noexcept
            /*!
                ensures
                    - heap storage pointer is moved
            !*/
            :   ptr{std::exchange(other.ptr, nullptr)},
                del{std::exchange(other.del, nullptr)},
                copy{std::exchange(other.copy, nullptr)},
                type_id{std::exchange(other.type_id, nullptr)}
            {
            }

            storage_heap& operator=(storage_heap&& other) noexcept
            /*!
                ensures
                    - if is_empty() == false then
                        - underlying object is destructed
                    - heap storage pointer is moved
            !*/
            {
                if (this != &other) 
                {
                    clear();
                    ptr     = std::exchange(other.ptr, nullptr);
                    del     = std::exchange(other.del, nullptr);
                    copy    = std::exchange(other.copy, nullptr);
                    type_id = std::exchange(other.type_id, nullptr);
                }
                return *this;
            }

            ~storage_heap()
            /*!
                ensures
                    - if is_empty() == false then
                        - underlying object is destructed
                    - is_empty() == true
            !*/
            {
                if (ptr)
                    del(ptr);
            }

            void clear()
            /*!
                ensures
                    - if is_empty() == false then
                        - underlying object is destructed
                    - is_empty() == true
            !*/
            {
                storage_heap{std::move(*this)};
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object
            !*/
            {
                return ptr;
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object
            !*/
            {
                return ptr;
            }

            void* ptr                     = nullptr;
            void  (*del)(void*)           = nullptr;
            void* (*copy)(const void*)    = nullptr;
            std::type_index (*type_id)()  = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

        template <std::size_t Size, std::size_t Alignment = 8>
        struct storage_stack : storage_base<storage_stack<Size, Alignment>>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.
                    This particular storage type uses stack allocation using a template size and alignment.
                    Therefore, only objects whose size and alignment fits the template parameters can be
                    erased and absorved into this object.
            !*/

            storage_stack() = default;
            /*!
                ensures
                    - this object is properly initialized
                    - is_empty() == true
                    - for all T: contains<T>() == false
            !*/

            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_stack, T> = true
            >
            storage_stack(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - Conversion constructor that copies or moves the incoming object (depending on the forwarding reference)
                    - is_empty() == true
                    - contains<std::decay_t<T>>() == true, otherwise contains<U>() == false for all other types
            !*/
            :   del{[](storage_stack& self) {
                    reinterpret_cast<T_*>(&self.data)->~T_();
                    self.del        = nullptr;
                    self.copy       = nullptr;
                    self.move       = nullptr;
                    self.type_id    = nullptr;
                }},
                copy{[](const storage_stack& src, storage_stack& dst) {
                    new (&dst.data) T_{*reinterpret_cast<const T_*>(&src.data)};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id = src.type_id;
                }},
                move{[](storage_stack& src, storage_stack& dst) {
                    new (&dst.data) T_{std::move(*reinterpret_cast<T_*>(&src.data))};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id = src.type_id;
                }},
                type_id{[] {
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
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor
                        - is_empty() == false
            !*/
            {
                if (other.copy)
                    other.copy(other, *this);
            }

            storage_stack& operator=(const storage_stack& other)
            /*!
                ensures
                    - underlying object is destructed if is_empty() == false
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor
                        - is_empty() == false
                    - else
                        - is_empty() == true
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
                    - if other.is_empty() == false then
                        - underlying object of other is moved using erased type's moved constructor
                        - is_empty() == false
            !*/
            {
                if (other.move)
                    other.move(other, *this);
            }

            storage_stack& operator=(storage_stack&& other)
            /*!
                ensures
                    - underlying object is destructed if is_empty() == false
                    - if other.is_empty() == false then
                        - underlying object of other is moved using erased type's moved constructor
                        - is_empty() == false
                    - else
                        - is_empty() == true
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
                    - calls clear()
            !*/
            {
                clear();
            }

            void clear()
            /*!
                ensures
                    - if is_empty() == false then
                        - underlying object is destructed
                    - is_empty() == true
            !*/
            {
                if (del)
                    del(*this);
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object
            !*/
            {
                return del ? (void*)&data : nullptr;
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object
            !*/
            {
                return del ? (const void*)&data : nullptr;
            }

            std::aligned_storage_t<Size, Alignment> data;
            void (*del)(storage_stack&)                         = nullptr;
            void (*copy)(const storage_stack&, storage_stack&)  = nullptr;
            void (*move)(storage_stack&, storage_stack&)        = nullptr;
            std::type_index (*type_id)()                        = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

        template <std::size_t Size, std::size_t Alignment = 8>
        struct storage_sbo : storage_base<storage_sbo<Size, Alignment>>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.
                    This particular storage type uses small buffer optimization (SBO), i.e. optional stack allocation
                    if the erased type fits the SBO parameters, otherwise, it uses heap allocation.
            !*/

            template<typename T_>
            struct type_fits : std::integral_constant<bool, sizeof(T_) <= Size && Alignment % alignof(T_) == 0>{};

            storage_sbo() = default;
            /*!
                ensures
                    - this object is properly initialized
                    - is_empty() == true
                    - for all T: contains<T>() == false
            !*/

            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_sbo, T> = true,
                std::enable_if_t<type_fits<T_>::value, bool> = true
            >
            storage_sbo(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - Conversion constructor that copies or moves the incoming object (depending on the forwarding reference)
                    - is_empty() == true
                    - contains<std::decay_t<T>>() == true, otherwise contains<U>() == false for all other types
                    - stack allocation is used
            !*/
            :   ptr{new (&data) T_{std::forward<T>(t)}},
                del{[](storage_sbo& self) {
                    reinterpret_cast<T_*>(&self.data)->~T_();
                    self.ptr        = nullptr;
                    self.del        = nullptr;
                    self.copy       = nullptr;
                    self.move       = nullptr;
                    self.type_id    = nullptr;
                }},
                copy{[](const storage_sbo& src, storage_sbo& dst) {
                    dst.ptr     = new (&dst.data) T_{*reinterpret_cast<const T_*>(src.ptr)};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id = src.type_id;
                }},
                move{[](storage_sbo& src, storage_sbo& dst) {
                    dst.ptr     = new (&dst.data) T_{std::move(*reinterpret_cast<T_*>(src.ptr))};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id = src.type_id;
                }},
                type_id{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_sbo, T> = true,
                std::enable_if_t<!type_fits<T_>::value, bool> = true
            >
            storage_sbo(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - Conversion constructor that copies or moves the incoming object (depending on the forwarding reference)
                    - is_empty() == true
                    - contains<std::decay_t<T>>() == true, otherwise contains<U>() == false for all other types
                    - heap allocation is used
            !*/
            :   ptr{new T_{std::forward<T>(t)}},
                del{[](storage_sbo& self) {
                    delete reinterpret_cast<T_*>(self.ptr);
                    self.ptr        = nullptr;
                    self.del        = nullptr;
                    self.copy       = nullptr;
                    self.move       = nullptr;
                    self.type_id    = nullptr;
                }},
                copy{[](const storage_sbo& src, storage_sbo& dst) {
                    dst.ptr     = new T_{*reinterpret_cast<const T_*>(src.ptr)};
                    dst.del     = src.del;
                    dst.copy    = src.copy;
                    dst.move    = src.move;
                    dst.type_id = src.type_id;
                }},
                move{[](storage_sbo& src, storage_sbo& dst) {
                    dst.ptr     = std::exchange(src.ptr,     nullptr);
                    dst.del     = std::exchange(src.del,     nullptr);
                    dst.copy    = std::exchange(src.copy,    nullptr);
                    dst.move    = std::exchange(src.move,    nullptr);
                    dst.type_id = std::exchange(dst.type_id, nullptr);
                }},
                type_id{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            storage_sbo(const storage_sbo& other)
            /*!
                ensures
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor
                        - is_empty() == false
            !*/
            {
                if (other.copy)
                    other.copy(other, *this);
            }

            storage_sbo& operator=(const storage_sbo& other)
            /*!
                ensures
                    - underlying object is destructed if is_empty() == false
                    - if other.is_empty() == false then
                        - underlying object of other is copied using erased type's copy constructor
                        - is_empty() == false
                    - else
                        - is_empty() == true
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
                    - if other.is_empty() == false then
                        - if underlying object of other is allocated on stack then
                            - underlying object of other is moved using erased type's moved constructor
                        - else
                            - storage heap pointer is moved
                        - is_empty() == false
            !*/
            {
                if (other.move)
                    other.move(other, *this);
            }

            storage_sbo& operator=(storage_sbo&& other)
            /*!
                ensures
                    - underlying object is destructed if is_empty() == false
                    - if other.is_empty() == false then
                        - if underlying object of other is allocated on stack then
                            - underlying object of other is moved using erased type's moved constructor
                        - else
                            - storage heap pointer is moved
                        - is_empty() == false
                    - else
                        - is_empty() == true
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
                    - calls clear()
            !*/
            {
                clear();
            }

            void clear()
            /*!
                ensures
                    - if is_empty() == false then
                        - underlying object is destructed
                    - is_empty() == true
            !*/
            {
                if (ptr)
                    del(*this);
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object
            !*/
            {
                return ptr;
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object
            !*/
            {
                return ptr;
            }

            std::aligned_storage_t<Size, Alignment> data;
            void* ptr                                       = nullptr;
            void (*del)(storage_sbo&)                       = nullptr;
            void (*copy)(const storage_sbo&, storage_sbo&)  = nullptr;
            void (*move)(storage_sbo&, storage_sbo&)        = nullptr;
            std::type_index (*type_id)()                    = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

        struct storage_shared : storage_base<storage_shared>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.
                    This particular storage type uses std::shared_ptr<void> to store and erase incoming
                    objects. Therefore, it uses heap allocation and reference counting.
            !*/

            storage_shared() = default;
            /*!
                ensures
                    - this object is properly initialized
                    - is_empty() == true
                    - for all T: contains<T>() == false
            !*/

            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_shared, T> = true
            >
            storage_shared(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            /*!
                ensures
                    - Conversion constructor that copies or moves the incoming object (depending on the forwarding reference)
                    - is_empty() == true
                    - contains<std::decay_t<T>>() == true, otherwise contains<U>() == false for all other types
            !*/
            :   ptr{std::make_shared<T_>(std::forward<T>(t))},
                type_id{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            storage_shared(const storage_shared& other)             = default;
            storage_shared& operator=(const storage_shared& other)  = default;

            storage_shared(storage_shared&& other) noexcept
            : ptr{std::move(other.ptr)},
              type_id{std::exchange(other.type_id, nullptr)}
            {
            }

            storage_shared& operator=(storage_shared&& other) noexcept
            {
                if (this != &other)
                {
                    ptr     = std::move(other.ptr);
                    type_id = std::exchange(other.type_id, nullptr);
                }
                    
                return *this;
            }

            void clear() 
            /*!
                ensures
                    - nulls the underlying shared_ptr<void>
                    - if this is the last reference then
                        - underlying object is destructed
                    - is_empty() == true
            !*/
            { 
                ptr     = nullptr;
                type_id = nullptr;
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object
            !*/
            {
                return ptr.get();
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object
            !*/
            {
                return ptr.get();
            }

            std::shared_ptr<void> ptr    = nullptr;
            std::type_index (*type_id)() = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

        struct storage_view : storage_base<storage_view>
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a storage type that uses type erasure to erase any type.
                    This particular storage type is a view type, similar to std::string_view or std::span.
                    So underlying objects are only ever referenced, not copied, moved or destructed.
            !*/

            storage_view() = default;
            /*!
                ensures
                    - this object is properly initialized
                    - is_empty() == true
                    - for all T: contains<T>() == false
            !*/
            
            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_view, T> = true
            >
            storage_view(T &&t) noexcept
            /*!
                ensures
                    - Conversion constructor that copies the address of the incoming object
                    - is_empty() == true
                    - contains<std::decay_t<T>>() == true, otherwise contains<U>() == false for all other types
            !*/
            :   ptr{&t},
                type_id{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }
            
            storage_view(const storage_view& other)             = default;
            storage_view& operator=(const storage_view& other)  = default;

            storage_view(storage_view&& other) noexcept
            : ptr{std::exchange(other.ptr, nullptr)},
              type_id{std::exchange(other.type_id, nullptr)}
            {
            }

            storage_view& operator=(storage_view&& other) noexcept
            {
                if (this != &other)
                {
                    ptr     = std::exchange(other.ptr, nullptr);
                    type_id = std::exchange(other.type_id, nullptr);
                }
                    
                return *this;
            }

            void clear() 
            /*!
                ensures
                    - nulls the underlying pointer
            !*/
            { 
                ptr     = nullptr;
                type_id = nullptr;
            }

            void* get_ptr()       
            /*!
                ensures
                    - returns a pointer to the underlying object
            !*/
            {
                return ptr;
            }

            const void* get_ptr() const 
            /*!
                ensures
                    - returns a const pointer to the underlying object
            !*/
            {
                return ptr;
            }

            void* ptr = nullptr;
            std::type_index (*type_id)() = nullptr;
        };

// -----------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_TYPE_ERASURE_H_
