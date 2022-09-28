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
    namespace te
    {
        template<class Storage, class T>
        using is_valid = std::enable_if_t<!std::is_same<std::decay_t<T>, Storage>::value, bool>;

        template<class Storage>
        struct storage_base
        {         
            bool is_empty() const
            {
                const Storage& me = *static_cast<const Storage*>(this);
                return me.get_ptr() == nullptr;
            }

            template<typename T>
            bool contains() const
            {
                const Storage& me = *static_cast<const Storage*>(this);
                return me.type_id ? me.type_id() == std::type_index{typeid(T)} : false;
            }

            template<typename T>
            T& unsafe_get() 
            {
                Storage& me = *static_cast<Storage*>(this);
                return *reinterpret_cast<T*>(me.get_ptr()); 
            }

            template<typename T>
            const T& unsafe_get() const 
            {
                const Storage& me = *static_cast<const Storage*>(this);
                return *reinterpret_cast<const T*>(me.get_ptr()); 
            }
        };

        struct storage_heap : storage_base<storage_heap>
        {
            storage_heap() = default;

            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_heap, T> = true
            >
            storage_heap(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
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
            :   ptr{other.ptr ? other.copy(other.ptr) : nullptr},
                del{other.del},
                copy{other.copy},
                type_id{other.type_id}
            {
            }

            storage_heap& operator=(const storage_heap& other)
            {
                if (this != &other) 
                    *this = std::move(storage_heap{other});
                return *this;
            }

            storage_heap(storage_heap&& other) noexcept
            :   ptr{std::exchange(other.ptr, nullptr)},
                del{std::exchange(other.del, nullptr)},
                copy{std::exchange(other.copy, nullptr)},
                type_id{std::exchange(other.type_id, nullptr)}
            {
            }

            storage_heap& operator=(storage_heap&& other) noexcept
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
            {
                if (ptr)
                    del(ptr);
            }

            void clear()
            {
                storage_heap{std::move(*this)};
            }

            void*       get_ptr()       {return ptr;}
            const void* get_ptr() const {return ptr;}

            void* ptr                     = nullptr;
            void  (*del)(void*)           = nullptr;
            void* (*copy)(const void*)    = nullptr;
            std::type_index (*type_id)()  = nullptr;
        };

        template <std::size_t Size, std::size_t Alignment = 8>
        struct storage_stack : storage_base<storage_stack<Size, Alignment>>
        {
            using mem_t = std::aligned_storage_t<Size, Alignment>;

            storage_stack() = default;

            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_stack, T> = true
            >
            storage_stack(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            :   del{[](mem_t& self) {
                    reinterpret_cast<T_*>(&self)->~T_();
                }},
                copy{[](const mem_t& self, mem_t& other) {
                    new (&other) T_{*reinterpret_cast<const T_*>(&self)};
                }},
                move{[](mem_t& self, mem_t& other) {
                    new (&other) T_{std::move(*reinterpret_cast<T_*>(&self))};
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
            :   del{other.del},
                copy{other.copy},
                move{other.move},
                type_id{other.type_id}
            {
                if (other.copy)
                    other.copy(other.data, data);
            }

            storage_stack& operator=(const storage_stack& other)
            {
                if (this != &other) 
                {
                    clear();
                    if (other.copy)
                        other.copy(other.data, data);
                    del     = other.del;
                    copy    = other.copy;
                    move    = other.move;
                    type_id = other.type_id;
                }
                return *this;
            }

            storage_stack(storage_stack&& other)
            :   del{other.del},
                copy{other.copy},
                move{other.move},
                type_id{other.type_id}
            {
                if (other.move)
                    other.move(other.data, data);
            }

            storage_stack& operator=(storage_stack&& other)
            {
                if (this != &other) 
                {
                    clear();
                    if (other.move)
                        other.move(other.data, data);
                    del     = other.del;
                    copy    = other.copy;
                    move    = other.move;
                    type_id = other.type_id;
                }
                return *this;
            }

            ~storage_stack()
            {
                clear();
            }

            void clear()
            {
                if (del)
                    del(data);
                del     = nullptr;
                copy    = nullptr;
                move    = nullptr;
                type_id = nullptr;
            }

            void*       get_ptr()       {return del ? (void*)&data : nullptr;}
            const void* get_ptr() const {return del ? (const void*)&data : nullptr;}

            mem_t data;
            void (*del)(mem_t&)                = nullptr;
            void (*copy)(const mem_t&, mem_t&) = nullptr;
            void (*move)(mem_t&, mem_t&)       = nullptr;
            std::type_index (*type_id)()       = nullptr;
        };

        template <std::size_t Size, std::size_t Alignment = 8>
        struct storage_sbo : storage_base<storage_sbo<Size, Alignment>>
        {
            template<typename T_>
            struct type_fits : std::integral_constant<bool, sizeof(T_) <= Size && Alignment % alignof(T_) == 0>{};

            using mem_t = std::aligned_storage_t<Size, Alignment>;

            storage_sbo() = default;

            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_sbo, T> = true,
                std::enable_if_t<type_fits<T_>::value, bool> = true
            >
            storage_sbo(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            :   ptr{new (&data) T_{std::forward<T>(t)}},
                del{[](mem_t& self_mem, void*) {
                    reinterpret_cast<T_*>(&self_mem)->~T_();
                }},
                copy{[](const void* self, mem_t& other) -> void* {
                    return new (&other) T_{*reinterpret_cast<const T_*>(self)};
                }},
                move{[](void*& self, mem_t& other) -> void* {
                    return new (&other) T_{std::move(*reinterpret_cast<T_*>(self))};
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
            :   ptr{new T_{std::forward<T>(t)}},
                del{[](mem_t&, void* self_ptr) {
                    delete reinterpret_cast<T_*>(self_ptr);
                }},
                copy{[](const void* self, mem_t&) -> void* {
                    return new T_{*reinterpret_cast<const T_*>(self)};
                }},
                move{[](void*& self, mem_t&) -> void* {
                    return std::exchange(self, nullptr);
                }},
                type_id{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            storage_sbo(const storage_sbo& other)
            :   ptr{other.ptr ? other.copy(other.ptr, data) : nullptr},
                del{other.del},
                copy{other.copy},
                move{other.move},
                type_id{other.type_id}
            {
            }

            storage_sbo& operator=(const storage_sbo& other)
            {
                if (this != &other) 
                {
                    clear();
                    ptr     = other.ptr ? other.copy(other.ptr, data) : nullptr;
                    del     = other.del;
                    copy    = other.copy;
                    move    = other.move;
                    type_id = other.type_id;
                }
                return *this;
            }

            storage_sbo(storage_sbo&& other)
            :   ptr{other.ptr ? other.move(other.ptr, data) : nullptr},
                del{other.del},
                copy{other.copy},
                move{other.move},
                type_id{other.type_id}
            {
            }

            storage_sbo& operator=(storage_sbo&& other)
            {
                if (this != &other) 
                {
                    clear();
                    ptr     = other.ptr ? other.move(other.ptr, data) : nullptr;
                    del     = other.del;
                    copy    = other.copy;
                    move    = other.move;
                    type_id = other.type_id;
                }
                return *this;
            }

            ~storage_sbo()
            {
                clear();
            }

            void clear()
            {
                if (ptr)
                    del(data, ptr);
                ptr     = nullptr;
                del     = nullptr;
                copy    = nullptr;
                move    = nullptr;
                type_id = nullptr;
            }

            void*       get_ptr()       {return ptr;}
            const void* get_ptr() const {return ptr;}

            mem_t data;
            void* ptr                           = nullptr;
            void  (*del)(mem_t&, void*)         = nullptr;
            void* (*copy)(const void*, mem_t&)  = nullptr;
            void* (*move)(void*&, mem_t&)       = nullptr;
            std::type_index (*type_id)()        = nullptr;
        };

        struct storage_shared : storage_base<storage_shared>
        {
            storage_shared() = default;
            
            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_shared, T> = true
            >
            storage_shared(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            :   ptr{std::make_shared<T_>(std::forward<T>(t))},
                type_id{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            void clear() { ptr = nullptr; }

            void*       get_ptr()       {return ptr.get();}
            const void* get_ptr() const {return ptr.get();}

            std::shared_ptr<void> ptr    = nullptr;
            std::type_index (*type_id)() = nullptr;
        };

        struct storage_view : storage_base<storage_view>
        {
            storage_view() = default;
            
            template <
                class T,
                class T_ = std::decay_t<T>,
                is_valid<storage_view, T> = true
            >
            storage_view(T &&t) noexcept
            :   ptr{&t},
                type_id{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }
            
            storage_view(const storage_view& other)             = default;
            storage_view& operator=(const storage_view& other)  = default;

            storage_view(storage_view&& other) noexcept
            : ptr{std::exchange(other.ptr, nullptr)}
            {
            }

            storage_view& operator=(storage_view&& other) noexcept
            {
                if (this != &other)
                    ptr = std::exchange(other.ptr, nullptr);
                return *this;
            }

            void clear() { ptr = nullptr; }

            void*       get_ptr()       {return ptr;}
            const void* get_ptr() const {return ptr;}

            void* ptr = nullptr;
            std::type_index (*type_id)() = nullptr;
        };
    }
}

#endif //DLIB_TYPE_ERASURE_H_
