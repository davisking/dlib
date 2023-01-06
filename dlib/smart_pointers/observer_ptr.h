#ifndef DLIB_OBSERVER_PTR_H
#define DLIB_OBSERVER_PTR_H

#include <utility>
#include <type_traits>

namespace dlib
{
    template<class T>
    class observer_ptr
    {
    public:
        using pointer   = std::add_pointer_t<T>;
        using reference = std::add_lvalue_reference_t<T>;

        constexpr observer_ptr()                                noexcept = default;
        constexpr observer_ptr(const observer_ptr&)             noexcept = default;
        constexpr observer_ptr& operator=(const observer_ptr&)  noexcept = default;

        constexpr observer_ptr(std::nullptr_t)      noexcept {};
        constexpr explicit observer_ptr(pointer p)  noexcept : ptr{p} {}

        template <
          class T2, 
          std::enable_if_t<std::is_convertible<std::add_pointer_t<T2>, pointer>::value, bool> = true
        >
        constexpr observer_ptr(observer_ptr<T2> p) noexcept : ptr{p.get()} {}

        constexpr observer_ptr(observer_ptr&& other) noexcept 
        : ptr{std::exchange(other.ptr, nullptr)} 
        {
        }

        constexpr observer_ptr& operator=(observer_ptr&& other) noexcept
        {
            if (this != &other)
                ptr = std::exchange(other.ptr, nullptr);
            return *this;
        }

        constexpr pointer   release()                     noexcept {return std::exchange(ptr, nullptr);}
        constexpr void      reset(pointer p = nullptr)    noexcept {ptr = p;}
        constexpr pointer   get()                   const noexcept {return ptr;}
        constexpr reference operator*()             const          {return *ptr;}
        constexpr pointer   operator->()            const noexcept {return ptr;}
        constexpr explicit operator bool()          const noexcept {return ptr != nullptr;}
        constexpr explicit operator pointer()       const noexcept {return ptr;}
        constexpr void swap(observer_ptr& p)              noexcept {std::swap(ptr, p.ptr);}

    private:
        T* ptr{nullptr};
    };

    template<class W1, class W2> bool operator==(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2) {return p1.get() == p2.get();}
    template<class W1, class W2> bool operator!=(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2) {return !(p1 == p2);}
    template<class W>            bool operator==(const observer_ptr<W>& p, std::nullptr_t) noexcept      {return !p;}
    template<class W>            bool operator==(std::nullptr_t, const observer_ptr<W>& p) noexcept      {return !p;}
    template<class W>            bool operator!=(const observer_ptr<W>& p, std::nullptr_t) noexcept      {return (bool)p;}
    template<class W>            bool operator!=(std::nullptr_t, const observer_ptr<W>& p) noexcept      {return (bool)p;}
}

#endif //DLIB_OBSERVER_PTR_H