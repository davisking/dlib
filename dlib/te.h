// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TYPE_ERASURE_H_
#define DLIB_TYPE_ERASURE_H_

#include <type_traits>
#include <utility>
#include <typeindex>

namespace dlib
{
    namespace te
    {
        class bad_te_cast : public std::bad_cast 
        {
        public:
            virtual const char * what() const throw()
            {
                return "bad_te_cast";
            }
        };

        struct storage_heap
        {
            storage_heap() = default;

            template <
                class T,
                class T_ = std::decay_t<T>,
                std::enable_if_t<!std::is_same<T_,storage_heap>::value, bool> = true
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

            template <
                class T,
                class T_ = std::decay_t<T>,
                std::enable_if_t<!std::is_same<T_,storage_heap>::value, bool> = true
            >
            storage_heap& operator=(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            {
                if (contains<T_>())
                    unsafe_get<T_>() = std::forward<T>(t);
                else
                    *this = std::move(storage_heap{std::forward<T>(t)});
                return *this;
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
                    storage_heap{std::move(*this)};
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

            bool is_empty() const
            {
                return ptr == nullptr;
            }

            template<typename T>
            bool contains() const
            {
                return ptr ? type_id() == std::type_index{typeid(T)} : false;
            }

            template<typename T>
            T& unsafe_get() {return *reinterpret_cast<T*>(ptr); }

            template<typename T>
            const T& unsafe_get() const {return *reinterpret_cast<const T*>(ptr); }

            void* ptr                     = nullptr;
            void  (*del)(void*)           = nullptr;
            void* (*copy)(const void*)    = nullptr;
            std::type_index (*type_id)()  = nullptr;
        };

        template <std::size_t Size, std::size_t Alignment = 8>
        struct storage_stack
        {
            using mem_t = std::aligned_storage_t<Size, Alignment>;

            storage_stack() = default;

            template <
                class T,
                class T_ = std::decay_t<T>,
                std::enable_if_t<!std::is_same<T_,storage_stack>::value, bool> = true
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

            template <
                class T,
                class T_ = std::decay_t<T>,
                std::enable_if_t<!std::is_same<T_,storage_stack>::value, bool> = true
            >
            storage_stack& operator=(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            {
                if (contains<T_>())
                    unsafe_get<T_>() = std::forward<T>(t);
                else
                    *this = std::move(storage_stack{std::forward<T>(t)});
                return *this;
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
                    *this = std::move(storage_stack{other});
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
                    storage_stack{std::move(*this)};
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
                if (del)
                    del(data);
            }

            void clear()
            {
                storage_stack{std::move(*this)};
            }

            bool is_empty() const
            {
                return del == nullptr;
            }

            template<typename T>
            bool contains() const
            {
                return del ? type_id() == std::type_index{typeid(T)} : false;
            }

            template<typename T>
            T& unsafe_get() {return *reinterpret_cast<T*>(&data); }

            template<typename T>
            const T& unsafe_get() const {return *reinterpret_cast<const T*>(&data); }

            mem_t data;
            void (*del)(mem_t&)                = nullptr;
            void (*copy)(const mem_t&, mem_t&) = nullptr;
            void (*move)(mem_t&, mem_t&)       = nullptr;
            std::type_index (*type_id)()       = nullptr;
        };

        template <std::size_t Size, std::size_t Alignment = 8>
        struct storage_sbo
        {
            template<typename T_>
            struct type_fits : std::integral_constant<bool, sizeof(T_) <= Size && Alignment % alignof(T_) == 0>{};

            using mem_t = std::aligned_storage_t<Size, Alignment>;

            storage_sbo() = default;

            template <
                class T,
                class T_ = std::decay_t<T>,
                std::enable_if_t<!std::is_same<T_,storage_sbo>::value, bool> = true,
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
                std::enable_if_t<!std::is_same<T_,storage_sbo>::value, bool> = true,
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

            template <
                class T,
                class T_ = std::decay_t<T>,
                std::enable_if_t<!std::is_same<T_,storage_sbo>::value, bool> = true
            >
            storage_sbo& operator=(T &&t) noexcept(std::is_nothrow_constructible<T_,T&&>::value)
            {
                if (contains<T_>())
                    unsafe_get<T_>() = std::forward<T>(t);
                else
                    *this = std::move(storage_sbo{std::forward<T>(t)});
                return *this;
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
                    *this = std::move(storage_sbo{other});
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
                    storage_sbo{std::move(*this)};
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
                if (ptr)
                    del(data, ptr);
            }

            void clear()
            {
                storage_sbo{std::move(*this)};
            }

            bool is_empty() const
            {
                return ptr == nullptr;
            }

            template<typename T>
            bool contains() const
            {
                return ptr ? type_id() == std::type_index{typeid(T)} : false;
            }

            template<typename T>
            T& unsafe_get() {return *reinterpret_cast<T*>(ptr); }

            template<typename T>
            const T& unsafe_get() const {return *reinterpret_cast<const T*>(ptr); }

            mem_t data;
            void* ptr                           = nullptr;
            void  (*del)(mem_t&, void*)         = nullptr;
            void* (*copy)(const void*, mem_t&)  = nullptr;
            void* (*move)(void*&, mem_t&)       = nullptr;
            std::type_index (*type_id)()        = nullptr;
        };

        struct storage_view
        {
            template <
                class T,
                class T_ = std::decay_t<T>,
                std::enable_if_t<!std::is_same<T_,storage_view>::value, bool> = true
            >
            storage_view(T &&t) noexcept
            :   ptr{&t},
                type_id{[] {
                    return std::type_index{typeid(T_)};
                }}
            {
            }

            void clear()
            {
                ptr = nullptr;
            }

            bool is_empty() const
            {
                return ptr == nullptr;
            }

            template<typename T>
            bool contains() const
            {
                return ptr ? type_id() == std::type_index{typeid(T)} : false;
            }

            template<typename T>
            T& unsafe_get() {return *reinterpret_cast<T*>(ptr); }

            template<typename T>
            const T& unsafe_get() const {return *reinterpret_cast<const T*>(ptr); }

            void* ptr = nullptr;
            std::type_index (*type_id)()        = nullptr;
        };
    }
}

#endif //DLIB_TYPE_ERASURE_H_