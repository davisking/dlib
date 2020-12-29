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
            upper bound of 5 dimensions (batch,channels,height,width,depth).
            All dimensions must be strictly positive.
         
            The object is either default constructed, constructed with an 
            initialiser list or with a pair of iterators

            If default-constructed, the object is empty and in an invalid state.
            That is, FFT functions will throw if attempted to be used with such 
            an object.

            If constructed with an initialiser list L, the object is properly
            initialised provided:
               - L.size() > 0 and L.size() <= 5
               - L contains strictly positive values
         
            If constructed with a pair of iterators, the behaviour of the 
            constructor is exactly the same as if constructed with an 
            initializer list spanned by those iterators.

            Once the object is constructed, it is immutable.
        !*/
    private:
        using container_type    = std::array<long,5>;
        using const_reference   = container_type::const_reference;
        using iterator          = container_type::iterator;
        using const_iterator    = container_type::const_iterator;
        
        size_t _size = 0;
        size_t _num_elements = 0;
        container_type _dims;

    public:
        fft_size() = default;
        /*!
            ensures
                - *this is properly initialised
                - size() == 0
        !*/
        
        fft_size(const long* dims_begin, const long* dims_end)
        {
            const size_t ndims = std::distance(dims_begin, dims_end);
            DLIB_ASSERT(ndims > 0, "the initialiser list must be non-empty");
            DLIB_ASSERT(ndims <= _dims.size(), "the initialiser list must have size less than 6");
            DLIB_ASSERT(std::find_if(dims_begin, dims_end, [](long dim) {return dim <= 0;}) == dims_end, "the initialiser list must contain strictly positive values");
            
            _num_elements = std::accumulate(dims_begin, dims_end, 1, std::multiplies<long>());
            
            if (_num_elements == 1)
            {
                _dims[0] = 1;
                _size = 1;
            }
            else
            {
                auto end = std::copy_if(dims_begin, dims_end, _dims.begin(), [](long dim) {return dim != 1;});
                _size = std::distance(_dims.begin(), end);
            }
        }
        /*!
            requires
                - std::distance(dims_begin, dims_end) > 0
                - std::distance(dims_begin, dims_end) <= 5
                - range contains strictly positive values
            ensures
                - *this is properly initialised
                - size() <= std::distance(dims_begin, dims_end)
                - num_elements() == product of all values in range
            throws
                - dlib::fatal_error if requirements aren't satisfied.
        !*/
        
        fft_size(std::initializer_list<long> dims)
        : fft_size(dims.begin(), dims.end())
        {
        }
        /*!
            requires
                - dims.size() > 0 and dims.size() <= 5
                - dims contains strictly positive values
            ensures
                - *this is properly initialised
                - size() <= dims.size()
                - num_elements() == product of all values in dims
            throws
                - dlib::fatal_error if requirements aren't satisfied.
        !*/
        
        size_t num_dims() const
        {
            return _size;
        }
        /*!
            ensures
                - returns the number of dimensions
                - if _size == 0, then *this is in an invalid state.
                  if _size > 0, then *this is in a valid state
        !*/
        
        long num_elements() const
        {
            return _num_elements;
        }
        /*!
            ensures
                - if _size > 0, returns the product of all dimensions, i.e. the total number
                  of elements
                - if _size == 0, returns 0
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
                - returns a const reference to the dimension at position index
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
                
        const_iterator begin() const
        {
            return _dims.begin();
        }
        /*!
            ensures
                - returns a const iterator that points to the first dimension 
                  in this container or end() if the array is empty.
        !*/
        
        const_iterator end() const
        {
            return _dims.begin() + _size;
        }
        /*!
            ensures
                - returns a const iterator that points to one past the end of 
                  the container.
        !*/
        
        bool operator==(const fft_size& other) const
        {
            return this->_size == other._size && std::equal(begin(), end(), other.begin());
        }
        /*!
            ensures
                - returns true if two fft_size objects have same size and same dimensions, i.e. if they have identical states
        !*/
    };
    
    inline dlib::uint32 hash(
        const fft_size& item,
        dlib::uint32 seed = 0)
    {
        seed = dlib::hash((dlib::uint64)item.num_dims(), seed);
        seed = std::accumulate(item.begin(), item.end(), seed, [](dlib::uint32 seed, long next) {
            return dlib::hash((dlib::uint64)next, seed);
        });
        return seed;
    }
    /*!
        ensures
            - returns a 32bit hash of the data stored in item.
    !*/

    inline fft_size pop_back(const fft_size& size)
    {
        DLIB_ASSERT(size.num_dims() > 0);
        return fft_size(size.begin(), size.end() - 1);
    }
    /*!
        requires
            - size.size() > 0
        ensures
            - returns a copy of size with the last dimension removed.
    !*/
}

#endif //DLIB_FFT_SIZE_H
