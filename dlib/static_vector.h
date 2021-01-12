// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_STATIC_VECTOR_H
#define DLIB_STATIC_VECTOR_H

#include <type_traits>
#include <iterator>
#include <numeric>
#include <utility>
#include <algorithm>
#include <array>
#include <exception>
#include <memory>
#include <functional>
#include <dlib/serialize.h>
#include <dlib/hash.h>

namespace dlib 
{    
    template <typename T, std::size_t Capacity>
    class static_vector
    {        
    public:
        using size_type = std::size_t;
        using value_type                = T;
        using reference                 = value_type&;
        using const_reference           = const value_type&;
        using pointer                   = value_type*;
        using const_pointer             = const value_type*;
        using iterator                  = value_type*;
        using const_iterator            = const value_type*;
        using reverse_iterator          = std::reverse_iterator<iterator>;
        using const_reverse_iterator    = std::reverse_iterator<const_iterator>;
        
        static_vector() noexcept 
        : m_size(0) 
        {}

        static_vector(size_type count) noexcept(noexcept(value_type{}))
        : m_size(0)
        {
            /*
             * Use resize_increase() instead of resize() to avoid calling
             * erase(), which requires move semantics
             */
            resize_increase(count);
        }

        static_vector(size_type count, const_reference value) noexcept(noexcept(value_type(value)))
        : m_size(0)
        {
            /*
             * Use resize_increase() instead of resize() to avoid calling
             * erase(), which requires move semantics
             */
            resize_increase(count, value);
        }

        static_vector(std::initializer_list<value_type> init_list)
        : m_size(init_list.size())
        {
            DLIB_ASSERT(m_size <= m_data.size());
            std::uninitialized_copy(init_list.begin(), init_list.end(), begin());
        }

        static_vector(const static_vector& other) 
        : m_size(other.m_size)
        {
            DLIB_ASSERT(m_size <= m_data.size());
            std::uninitialized_copy(other.begin(), other.end(), begin());
        }

        static_vector& operator=(const static_vector& other)
        {
            if (&other != this)
            {
                clear();
                m_size = other.m_size;
                std::uninitialized_copy(other.begin(), other.end(), begin());
            }
            return *this;
        }

        static_vector(static_vector&& other)
        : m_size(0)
        {
            swap(other);
        }

        static_vector& operator=(static_vector&& other) 
        {
            if (&other != this)
            {
                /* Technically don't need the following statement.
                 * The standard says other is left in a valid but otherwise 
                 * indeterminate state. Basically other can be whatever 
                 * we want it to be. Probably a good idea to render 
                 * other in an empty state*/
                clear();
                swap(other);
            }
            return *this;
        }

        ~static_vector() 
        { 
            clear(); 
        }
        
        void swap(static_vector& other)
        {
            std::swap(m_size, other.m_size);
            std::swap(m_data, other.m_data);
        }
        
        void resize(size_type count)
        {
            if (count < size())
            {
                erase(begin() + count, end());
            }
            else if (count > size())
            {
                resize_increase(count);
            }
        }
        
        void resize(size_type count, const_reference value)
        {
            if (count < m_size)
            {
                erase(begin() + count, end());
            }
            else if (count > m_size)
            {
                resize_increase(count,value);
            }
        }
        
        void push_back(const value_type& value) 
        {
            insert(end(), value);
        }
        
        void push_back(value_type&& value)
        {
            insert(end(), std::move(value));
        }
        
        template <typename... CtorArgs>
        void emplace_back(CtorArgs&&... args)
        {
            if (full())
                throw std::out_of_range("size()");
            new (storage_end()) value_type(std::forward<CtorArgs>(args)...);
            m_size++;
        }
        
        iterator insert(const_iterator pos, size_type count, const_reference value) 
        {
            if (m_data.size() < m_size + count)
                throw std::out_of_range("count");

            iterator mut_pos = const_cast<iterator>(pos);
            std::move_backward(mut_pos, end(), end() + count);
            std::uninitialized_fill(storage_begin() + (mut_pos - begin()), 
                                    storage_begin() + (mut_pos - begin()) + count, 
                                    value);
            m_size += count;
            return mut_pos;
        }
        
        iterator insert(const_iterator pos, const_reference value) 
        {
            return insert(pos, 1, value);
        }
        
        iterator insert(const_iterator pos, value_type&& value)
        {
            if (full())
                throw std::out_of_range("size()");
            iterator mut_pos = const_cast<iterator>(pos);
            std::move_backward(mut_pos, end(), end() + 1);
            new (mut_pos) value_type(std::move(value));
            m_size++;
            return mut_pos;
        }

        template <typename... CtorArgs>
        iterator emplace(const_iterator pos, CtorArgs&&... args) 
        {
            if (full())
                throw std::out_of_range("size()");
            iterator mut_pos = const_cast<iterator>(pos);
            std::move_backward(mut_pos, end(), end() + 1);
            new (mut_pos) value_type(std::forward<CtorArgs>(args)...);
            m_size++;
            return mut_pos;
        }

        iterator erase(const_iterator erase_begin, const_iterator erase_end) 
        {
            const auto nremoved = std::distance(erase_begin, erase_end);
            iterator mut_begin  = const_cast<iterator>(erase_begin);
            iterator mut_end    = const_cast<iterator>(erase_end);
            std::for_each(
                mut_begin,
                mut_end,
                [&](reference r) { r.~value_type(); });
            std::move(mut_end, end(), mut_begin);
            m_size -= nremoved;
            return mut_begin;
        }
        
        iterator erase(const_iterator pos) 
        {
            return erase(pos, pos + 1);
        }
        
        void clear() 
        {
            /* 
             * Don't use erase(begin(),end()) since 
             * that requires move semantics.
             */
            std::for_each(begin(), end(), [&](reference r) { r.~value_type(); });
            m_size = 0;
        }

        void pop_back()
        {
            erase(end() - 1);
        }
        
        void pop_front()
        {
            erase(begin());
        }
        
        reference at(size_t index) 
        {
            if (index >= m_size)
                throw std::out_of_range("index out of range");
            return data(index); 
        }
        const_reference at(size_t index) const
        { 
            if (index >= m_size)
                throw std::out_of_range("index out of range");
            return data(index);
        }
        
        reference       operator[](size_t index) noexcept { return data(index); }
        const_reference operator[](size_t index) const noexcept { return data(index);}

        reference       front() noexcept { return data(0); }
        const_reference front() const noexcept { return data(0); }

        reference       back() noexcept { return data(m_size - 1); }
        const_reference back() const noexcept { return data(m_size - 1); }

        pointer         data() noexcept {return reinterpret_cast<pointer>(&m_data[0]); }
        const_pointer   data() const noexcept {return reinterpret_cast<const_pointer>(&m_data[0]);}

        iterator        begin() noexcept { return data();}
        const_iterator  begin() const noexcept { return data(); }
        const_iterator  cbegin() const noexcept { return data(); }

        iterator        end() noexcept { return data() + m_size; }
        const_iterator  end() const noexcept { return data() + m_size; }
        const_iterator  cend() const noexcept { return data() + m_size; }

        reverse_iterator        rbegin() noexcept { return data() + m_size; }
        const_reverse_iterator  rbegin() const noexcept { return data() + m_size; }
        const_reverse_iterator  crbegin() const noexcept { return data() + m_size; }

        reverse_iterator        rend() noexcept { return data(); }
        const_reverse_iterator  rend() const noexcept { return data(); }
        const_reverse_iterator  crend() const noexcept { return data(); }

        constexpr size_type capacity() const noexcept { return m_data.size(); }
        size_type size() const noexcept { return m_size; }
        bool empty() const noexcept { return m_size == 0; }
        bool full() const noexcept { return m_size == m_data.size(); }
        
    private:
        using storage_type = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

        std::array<storage_type, Capacity> m_data;
        size_type m_size = 0;

        reference data(size_t index) noexcept 
        {
            return *reinterpret_cast<pointer>(&m_data[index]);
        }
        
        const_reference data(size_t index) const noexcept 
        {
            return *reinterpret_cast<const_pointer>(&m_data[index]);
        }
        
        void resize_increase(size_type count)
        {
            DLIB_ASSERT(count <= m_data.size());
            const auto nadditional = count - size();
            std::for_each(storage_end(), storage_end() + nadditional, [](storage_type& store) {
                new (static_cast<void*>(&store)) value_type;
            });
            m_size = count;
        }
        
        void resize_increase(size_type count, const_reference value)
        {
            DLIB_ASSERT(count <= m_data.size());
            const auto nadditional = count - size();
            std::uninitialized_fill(end(), end() + nadditional, value);
            m_size = count;
        }

        storage_type* storage_begin() noexcept { return &m_data[0]; }
        storage_type* storage_end() noexcept { return &m_data[m_size]; }
    };
    
    template <typename T, std::size_t Capacity>
    inline void swap (
        static_vector<T,Capacity>& a, 
        static_vector<T,Capacity>& b 
    ) { a.swap(b); }
    
    template <typename T, std::size_t Capacity>
    void serialize (
        const static_vector<T,Capacity>& item,  
        std::ostream& out
    )
    {
        try
        {
            serialize(item.capacity(),out);
            serialize(item.size(),out);
            for (const auto& x : item)
                serialize(x,out);
        }
        catch (serialization_error& e)
        { 
            throw serialization_error(e.info + "\n   while serializing object of type static_vector"); 
        }
    }

    template <typename T, std::size_t Capacity>
    void deserialize (
        static_vector<T,Capacity>& item,  
        std::istream& in
    )
    {
        try
        {
            std::size_t capacity, size;
            deserialize(capacity,in);
            if (capacity != item.capacity())
                throw serialization_error("item capacity doesn't match serialised value");
            deserialize(size,in);
            item.resize(size);
            for (auto& x : item)
                deserialize(x,in);
        }
        catch (serialization_error& e)
        { 
            item.clear();
            throw serialization_error(e.info + "\n   while deserializing object of type static_vector"); 
        }
    }
    
    #if __cplusplus < 201703L
    template<typename...> using void_t = void;
    #else
    using std::void_t;
    #endif

    template<typename T, typename dummy>
    using std_hash_helper = T;
    
    inline void hash_combine(std::size_t& seed) { }

    template <typename T, typename... Rest>
    inline void hash_combine(std::size_t& seed, const T& v, Rest... rest)
    {
        seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        hash_combine(seed, rest...);
    }

    template <typename T, typename = void>
    struct is_dlib_hashable : std::false_type {};
    
    template <typename T>
    struct is_dlib_hashable<T, void_t<decltype(dlib::hash(std::declval<const T&>(), std::declval<dlib::uint32>()))>> : std::true_type {};
    
    template <typename T, typename = void>
    struct is_std_hashable : std::false_type {};

    template <typename T>
    struct is_std_hashable<T, void_t<decltype(std::declval<std::hash<T>>()(std::declval<const T&>())),
                                     decltype(std::declval<std::hash<T>>()(std::declval<T const&>()))>> : std::true_type {};

    template <
        typename T, 
        std::size_t Capacity,
        typename std::enable_if<dlib::is_dlib_hashable<T>::value>::type* = nullptr
        >
    inline dlib::uint32 hash(
        const static_vector<T,Capacity>& item,
        dlib::uint32 seed = 0)
    {
        seed = dlib::hash((dlib::uint64)item.capacity(), seed);
        seed = dlib::hash((dlib::uint64)item.size(), seed);
        seed = std::accumulate(item.begin(), item.end(), seed, [](dlib::uint32 seed, const T& next) {
            return dlib::hash(next, seed);
        });
        return seed;
    }
}

namespace std 
{
    template <
        typename T, 
        std::size_t Capacity
        >
    struct hash<dlib::std_hash_helper<dlib::static_vector<T,Capacity>, typename std::enable_if<dlib::is_std_hashable<T>::value>::type>>
    {
        std::hash<T> hasher;
        std::size_t operator()(const dlib::static_vector<T,Capacity>& item) const 
        {
            std::size_t seed = 0;
            dlib::hash_combine(seed, item.capacity(), item.size());
            for (const auto& x : item)
                seed ^= hasher(x) + 0x9e3779b9 + (seed<<6) + (seed>>2);
            return seed;
        }
    };
}

#endif // DLIB_STATIC_VECTOR_H
