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
        /*!
            WHAT THIS OBJECT REPRESENTS
            This object is a container used to store the dimensions of an FFT 
            operation. It is implemented as a stack-based container with an 
            upper bound of 5 elements (batch,channels,height,width,depth).
            All elements must be strictly positive.
         
            The object is either default constructed or constructed with an 
            initialiser list.

            If default-constructed, the object is empty and in an invalid state.
            That is, FFT functions will throw if attempted to be used with such 
            an object.

            If constructed with an initialiser list L, the object is properly
            initialised provided:
               - L.size() > 0 and L.size() <= 5
               - L contains strictly positive values

            Once the object is constructed, it is immutable.
        !*/
    private:
        using container_type    = std::array<long,5>;
        using const_reference   = container_type::const_reference;
        using iterator          = container_type::iterator;
        using const_iterator    = container_type::const_iterator;
        
        size_t _size = 0;
        container_type _dims;
        
        iterator begin()
        {
            return _size > 0 ? _dims.begin() : nullptr;
        }
        
        const_iterator begin() const
        {
            return _size > 0 ? _dims.begin() : nullptr;
        }
        
        iterator end()
        {
            return _size > 0 ? _dims.begin() + _size : nullptr;
        }
        
        const_iterator end() const
        {
            return _size > 0 ? _dims.begin() + _size : nullptr;
        }

    public:
        fft_size() = default;
        /*!
            ensures
                - this is properly initialised
                - size() == 0
                - is_valid() == false
        !*/
        
        fft_size(std::initializer_list<long> dims)
        {
            DLIB_ASSERT(dims.size() > 0, "the initialiser list must be non-empty");
            DLIB_ASSERT(dims.size() <= _dims.size(), "the initialiser list must have size less than 6");
            std::copy(dims.begin(), dims.end(), _dims.begin());
            _size = dims.size();
            DLIB_ASSERT(is_valid(), "the initialiser list must contain strictly positive values");
        }
        /*!
            requires
                - dims.size() > 0 and dims.size() <= 5
                - dims contains strictly positive values
            ensures
                - *this is properly initialised
                - size() == dims.size()
                - is_valid() == true 
            throws
                - dlib::fatal_error if requirements aren't satisfied.
        !*/
        
        size_t size() const
        {
            return _size;
        }
        /*!
            ensures
                - returns the number of elements in *this
        !*/

        const_reference operator[](size_t index) const
        {
            DLIB_ASSERT(index < _size, "index " << index << " out of range [0," << _size << ")");
            return _dims[index];
        }
        /*!
            requires
                - size() > 0
                - index < size()
            ensures
                - returns a const reference to the element at position index
        !*/
        
        const_reference back() const
        {
            DLIB_ASSERT(_size > 0, "object is empty");
            return _dims[_size-1];
        }
        /*!
            requires
                - size() > 0
            ensures
                - returns a const reference to (*this)[size()-1]
        !*/
        
        long dimprod() const
        {
            DLIB_ASSERT(_size > 0, "object is empty");
            return std::accumulate(begin(), end(), 1, std::multiplies<long>());
        }
        /*!
            requires
                - size() > 0
            ensures
                - returns the product of all dimensions, i.e. the total number
                  of elements.
        !*/
        
        bool operator==(const fft_size& other) const
        {
            return this->_size == other._size && std::equal(begin(), end(), other.begin());
        }
        /*!
            ensures
                - returns true if two fft_size objects have same size and same elements.
        !*/
        
        bool is_valid() const
        {
            return _size > 0 && std::find_if(begin(), end(), [](long dim) {return dim <= 0;}) == end();
        }
        /*!
            ensures
                - returns true if:
                    - size() > 0
                    - (*this) contains strictly positive elements
        !*/
        
        /*global helpers*/
        friend inline dlib::uint32 hash(
            const fft_size& item,
            dlib::uint32 seed = 0)
        {
            seed = dlib::hash((dlib::uint64)item.size(), seed);
            seed = std::accumulate(item.begin(), item.end(), seed, [](dlib::uint32 seed, long next) {
                return dlib::hash((dlib::uint64)next, seed);
            });
            return seed;
        }
        /*!
            ensures
                - returns a 32bit hash of the data stored in item.
        !*/
        
        friend inline fft_size squeeze_ones(const fft_size& size)
        {
            fft_size newsize;
            if (size.dimprod() == 1)
            {
                newsize = {1};
            }
            else
            {
                newsize = size;
                const auto newend = std::remove(newsize.begin(), newsize.end(), 1);
                const long nremoved = std::distance(newend, newsize.end());
                newsize._size -= nremoved;
            }
            return newsize;
        }
        /*!
            ensures
                - returns a copy of size with all but 1 elements equal to 1 removed
        !*/
        
        friend inline fft_size pop_back(const fft_size& size)
        {
            DLIB_ASSERT(size.size() > 0);
            fft_size newsize = size;
            newsize._size--;
            return newsize;
        }
        /*!
            requires
                - size.size() > 0
            ensures
                - returns a copy of size with the last element removed.
        !*/
        
        /*global helpers*/
    };
}

#endif //DLIB_FFT_SIZE_H
