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

        const_reference operator[](size_t index) const
        {
            DLIB_ASSERT(index < _size);
            return _dims[index];
        }
        
        const_reference back() const
        {
            DLIB_ASSERT(_size > 0);
            return _dims[_size-1];
        }

        long dimprod() const
        {
            return std::accumulate(_dims.begin(), _dims.begin() + _size, 1, std::multiplies<long>());
        }

        bool operator==(const fft_size& other) const
        {
            return this->_size == other._size && std::equal(_dims.begin(), _dims.begin() + _size, other._dims.begin());
        }
        
        /*global helpers*/
        friend inline dlib::uint32 hash(
            const fft_size& item,
            dlib::uint32 seed = 0)
        {
            seed = dlib::hash((dlib::uint64)item._size, seed);
            seed = std::accumulate(item._dims.begin(), item._dims.begin() + item._size, seed, [](dlib::uint32 seed, long next) {
                return dlib::hash((dlib::uint64)next, seed);
            });
            return seed;
        }
        
        friend inline fft_size squeeze_ones(const fft_size& size)
        {
            fft_size newsize = size;
            const auto newend = std::remove(newsize._dims.begin(), newsize._dims.begin() + newsize._size, 1);
            const long nremoved = std::distance(newend, newsize._dims.begin() + newsize._size);
            newsize._size -= nremoved;
            return newsize;
        }
        
        friend inline fft_size pop_back(const fft_size& size)
        {
            fft_size newsize = size;
            newsize._size--;
            return newsize;
        }
        /*global helpers*/
    };
}

#endif //DLIB_FFT_SIZE_H
