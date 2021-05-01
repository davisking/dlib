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
    public:
        using container_type    = std::array<long,5>;
        using const_reference   = container_type::const_reference;
        using iterator          = container_type::iterator;
        using const_iterator    = container_type::const_iterator;
        
        fft_size() = default;
        /*!
            ensures
                - *this is properly initialised
                - num_dims() == 0
        !*/
        
        template<typename ConstIterator>
        fft_size(ConstIterator dims_begin, ConstIterator dims_end)
        /*!
            requires
                - ConstIterator is const iterator type that points to a long object
                - std::distance(dims_begin, dims_end) > 0
                - std::distance(dims_begin, dims_end) <= 5
                - range contains strictly positive values
            ensures
                - *this is properly initialised
                - num_dims() == std::distance(dims_begin, dims_end)
                - num_elements() == product of all values in range
        !*/
        {
            const size_t ndims = std::distance(dims_begin, dims_end);
            DLIB_ASSERT(ndims > 0, "fft_size objects must be non-empty");
            DLIB_ASSERT(ndims <= _dims.size(), "fft_size objects must have size less than 6");
            DLIB_ASSERT(std::find_if(dims_begin, dims_end, [](long dim) {return dim <= 0;}) == dims_end, "fft_size objects must contain strictly positive values");
            
            std::copy(dims_begin, dims_end, _dims.begin());
            _size = ndims;
            _num_elements = std::accumulate(dims_begin, dims_end, 1, std::multiplies<long>());
        }
        
        fft_size(std::initializer_list<long> dims)
        : fft_size(dims.begin(), dims.end())
        /*!
            requires
                - dims.size() > 0 and dims.size() <= 5
                - dims contains strictly positive values
            ensures
                - *this is properly initialised
                - num_dims() == dims.size()
                - num_elements() == product of all values in dims
        !*/
        {
        }
        
        size_t num_dims() const
        /*!
            ensures
                - returns the number of dimensions
        !*/
        {
            return _size;
        }
        
        long num_elements() const
        /*!
            ensures
                - if num_dims() > 0, returns the product of all dimensions, i.e. the total number
                  of elements
                - if num_dims() == 0, returns 0
        !*/
        {
            return _num_elements;
        }

        const_reference operator[](size_t index) const
        /*!
            requires
                - index < num_dims()
            ensures
                - returns a const reference to the dimension at position index
        !*/
        {
            DLIB_ASSERT(index < _size, "index " << index << " out of range [0," << _size << ")");
            return _dims[index];
        }
        
        const_reference back() const
        /*!
            requires
                - num_dims() > 0
            ensures
                - returns a const reference to (*this)[num_dims()-1]
        !*/
        {
            DLIB_ASSERT(_size > 0, "object is empty");
            return _dims[_size-1];
        }
                
        const_iterator begin() const
        /*!
            ensures
                - returns a const iterator that points to the first dimension 
                  in this container or end() if the array is empty.
        !*/
        {
            return _dims.begin();
        }
        
        const_iterator end() const
        /*!
            ensures
                - returns a const iterator that points to one past the end of 
                  the container.
        !*/
        {
            return _dims.begin() + _size;
        }
        
        bool operator==(const fft_size& other) const
        /*!
            ensures
                - returns true if two fft_size objects have same size and same dimensions, i.e. if they have identical states
        !*/
        {
            return this->_size == other._size && std::equal(begin(), end(), other.begin());
        }
        
    private:        
        size_t _size = 0;
        size_t _num_elements = 0;
        container_type _dims{};
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
            - num_dims.size() > 0
        ensures
            - returns a copy of size with the last dimension removed.
    !*/
    
    inline fft_size squeeze_ones(const fft_size size)
    {
        DLIB_ASSERT(size.num_dims() > 0);
        fft_size newsize;
        if (size.num_elements() == 1)
        {
            newsize = {1};
        }
        else
        {
            fft_size::container_type tmp;
            auto end = std::copy_if(size.begin(), size.end(), tmp.begin(), [](long dim){return dim != 1;});
            newsize = fft_size(tmp.begin(), end);
        }
        return newsize;
    }
    /*!
        requires
            - num_dims.size() > 0
        ensures
            - removes dimensions with values equal to 1, yielding a new fft_size object with the same num_elements() but fewer dimensions
    !*/
}

#endif //DLIB_FFT_SIZE_H
