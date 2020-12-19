#ifndef DLIB_FFT_SIZE_H
#define DLIB_FFT_SIZE_H

#include <array>
#include <algorithm>
#include <numeric>
#include "../assert.h"
#include "../hash.h"

namespace dlib
{
    class fft_size
    {   
    private:
        using container_type    = std::array<long,5>;
        using reference         = container_type::reference;
        using const_reference   = container_type::const_reference;

        size_t _size = 0;
        container_type _dims;

    public:
        fft_size() = default;
        
        fft_size(std::initializer_list<long> dims)
        {
            DLIB_ASSERT(dims.size() <= _dims.size());
            std::copy(dims.begin(), dims.end(), _dims.begin());
            _size = dims.size();
        }

        size_t size() const
        {
            return _size;
        }

        reference operator[](size_t index)
        {
            DLIB_ASSERT(index < _size);
            return _dims[index];
        }

        const_reference operator[](size_t index) const
        {
            DLIB_ASSERT(index < _size);
            return _dims[index];
        }
        
        reference back()
        {
            DLIB_ASSERT(_size > 0);
            return _dims[_size-1];
        }
        
        const_reference back() const
        {
            DLIB_ASSERT(_size > 0);
            return _dims[_size-1];
        }

        void pop_back()
        {
            if (_size > 0)
                _size--;
        }

        long dimprod() const
        {
            return std::accumulate(_dims.begin(), _dims.begin() + _size, 1, std::multiplies<long>());
        }

        void remove_ones()
        {
            const auto newend = std::remove(_dims.begin(), _dims.begin() + _size, 1);
            const long nremoved = std::distance(newend, _dims.begin() + _size);
            _size -= nremoved;
        }
        
        bool operator==(const fft_size& other) const
        {
            return this->_size == other._size && std::equal(_dims.begin(), _dims.begin() + _size, other._dims.begin());
        }
        
        friend inline uint32 hash(
            const fft_size& item,
            uint32 seed = 0)
        {
            seed = dlib::hash((uint64)item._size, seed);
            seed = std::accumulate(item._dims.begin(), item._dims.begin() + item._size, seed, [](uint32 seed, long next) {
                return dlib::hash((uint64)next, seed);
            });
            return seed;
        }
    };
}

#endif //DLIB_FFT_SIZE_H