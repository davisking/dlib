// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_TENSOR_ABSTRACT_H_
#ifdef DLIB_DNn_TENSOR_ABSTRACT_H_

#include "../matrix.h"
#include "../any/any_abstract.h"

namespace dlib
{
// ----------------------------------------------------------------------------------------

    class tensor
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a 4D array of float values, all stored contiguously
                in memory.  Importantly, it keeps two copies of the floats, one on the host
                CPU side and another on the GPU device side. It automatically performs the
                necessary host/device transfers to keep these two copies of the data in
                sync.

                All transfers to the device happen asynchronously with respect to the
                default CUDA stream so that CUDA kernel computations can overlap with data
                transfers.  However, any transfers from the device to the host happen
                synchronously in the default CUDA stream.  Therefore, you should perform
                all your CUDA kernel launches on the default stream so that transfers back
                to the host do not happen before the relevant computations have completed.

                If DLIB_USE_CUDA is not #defined then this object will not use CUDA at all.
                Instead, it will simply store one host side memory block of floats.  

                Finally, the convention in dlib code is to interpret the tensor as a set of
                num_samples() 3D arrays, each of dimension k() by nr() by nc().  Also,
                while this class does not specify a memory layout, the convention is to
                assume that indexing into an element at coordinates (sample,k,nr,nc) can be
                accomplished via:
                    host()[((sample*t.k() + k)*t.nr() + nr)*t.nc() + nc]

            THREAD SAFETY
                Instances of this object are not thread-safe.  So don't touch one from
                multiple threads at the same time.
        !*/

    public:

        virtual ~tensor();

        long num_samples(
        ) const; 
        /*!
            ensures
                - returns the number of 3D arrays of dimension k() by nr() by nc() there
                  are in this object.  
        !*/

        long k(
        ) const; 
        /*!
            ensures
                - returns the k dimension of this tensor.  Generally, we think of a tensor
                  as containing num_samples() images of nr() by nc() rows and columns, each
                  with k() channels.
        !*/

        long nr(
        ) const; 
        /*!
            ensures
                - returns the number of rows in this tensor.
        !*/

        long nc(
        ) const; 
        /*!
            ensures
                - returns the number of columns in this tensor.
        !*/

        size_t size(
        ) const;
        /*!
            ensures
                - returns num_samples()*k()*nr()*nc()
                  (i.e. the total number of floats in this tensor)
        !*/

        void async_copy_to_device(
        ) const;
        /*!
            ensures
                - This function does not block.
                - if (the host version of the data is newer than the device's copy) then
                    - Begins asynchronously copying host data to the device.
                    - A call to device() that happens before the transfer completes will
                      block until the transfer is complete.  That is, it is safe to call
                      async_copy_to_device() and then immediately call device().
        !*/

        typedef float* iterator;
        typedef const float* const_iterator;
        iterator       begin()       { return host(); }
        const_iterator begin() const { return host(); }
        iterator       end()         { return host()+size(); }
        const_iterator end() const   { return host()+size(); }
        /*!
            ensures
                - makes a tensor iterable just like the STL containers.   
        !*/

        virtual const float* host(
        ) const = 0;
        /*!
            ensures
                - returns a pointer to the host memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (the host's copy of the data is out of date) then
                    - copies the data from the device to the host, while this is happening
                      the call to host() blocks. 
        !*/

        virtual float* host(
        ) = 0;
        /*!
            ensures
                - returns a pointer to the host memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (the host's copy of the data is out of date) then
                    - copies the data from the device to the host, while this is happening
                      the call to host() blocks. 
                - Marks the device side data as out of date so that the next call to
                  device() will perform a host to device transfer.  If you want to begin
                  the transfer immediately then you can call async_copy_to_device() after
                  calling host().
        !*/

        float float* host_write_only(
        ) = 0;
        /*!
            ensures
                - This function returns the same pointer as host(), except that it never
                  performs a device to host memory copy.  Instead, it immediately marks the
                  device side data as out of date, effectively discarding it.  Therefore,
                  the values in the data pointed to by host_write_only() are undefined and
                  you should only call host_write_only() if you are going to assign to
                  every memory location in the returned memory block.  
        !*/

        virtual const float* device(
        ) const = 0;
        /*!
            requires
                - DLIB_USE_CUDA is #defined
            ensures
                - returns a pointer to the device memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (the device's copy of the data is out of date) then
                    - copies the data from the host to the device, while this is happening
                      the call to device() blocks. 
        !*/

        virtual float* device(
        ) = 0;
        /*!
            requires
                - DLIB_USE_CUDA is #defined
            ensures
                - returns a pointer to the device memory block of size() contiguous float
                  values or nullptr if size()==0.
                - if (the device's copy of the data is out of date) then
                    - copies the data from the host to the device, while this is happening
                      the call to device() blocks. 
                - Marks the host side data as out of date so that the next call to
                  host() will perform a device to host transfer.
        !*/

        float float* device_write_only(
        ) = 0;
        /*!
            requires
                - DLIB_USE_CUDA is #defined
            ensures
                - This function returns the same pointer as device(), except that it never
                  performs a host to device memory copy.  Instead, it immediately marks the
                  host side data as out of date, effectively discarding it.  Therefore, the
                  values in the data pointed to by device_write_only() are undefined and
                  you should only call device_write_only() if you are going to assign to
                  every memory location in the returned memory block.  
        !*/

        virtual const any& annotation(
        ) const = 0;
        /*!
            ensures
                - returns a const reference to the any object in this tensor.  The any
                  object can be used to store any additional annotation you like in a
                  tensor.  However, it should be noted that the annotation() is ignored by
                  serialize() and therefore not saved when a tensor is serialized.
        !*/

        virtual any& annotation(
        ) = 0;
        /*!
            ensures
                - returns a non-const reference to the any object in this tensor.  The any
                  object can be used to store any additional annotation you like in a
                  tensor.  However, it should be noted that the annotation() is ignored by
                  serialize() and therefore not saved when a tensor is serialized.
        !*/

        int device_id(
        ) const; 
        /*!
            ensures
                - returns the ID of the CUDA device that allocated this memory. I.e. the
                  number returned by cudaGetDevice() when the memory was allocated.
                - If CUDA is not being used then this function always returns 0.
        !*/

        tensor& operator= (
            float val
        );
        /*!
            ensures
                - sets all elements of this tensor equal to val.
                - returns *this
        !*/

        tensor& operator*= (
            float val
        );
        /*!
            ensures
                - pointwise multiplies all elements of *this tensor with val.
                - returns *this
        !*/
        
        tensor& operator/= (
            float val
        );
        /*!
            ensures
                - pointwise divides all elements of *this tensor with val.
                - returns *this
        !*/

        template <typename EXP>
        tensor& operator= (
            const matrix_exp<EXP>& item
        );
        /*!
            requires
                - num_samples() == item.nr()
                - k()*nr()*nc() == item.nc()
                - item contains float values
            ensures
                - Assigns item to *this tensor by performing:
                  set_ptrm(host(), num_samples(), k()*nr()*nc()) = item;
        !*/

        template <typename EXP>
        tensor& operator+= (
            const matrix_exp<EXP>& item
        );
        /*!
            requires
                - num_samples() == item.nr()
                - k()*nr()*nc() == item.nc()
                - item contains float values
            ensures
                - Adds item to *this tensor by performing:
                  set_ptrm(host(), num_samples(), k()*nr()*nc()) += item;
        !*/

        template <typename EXP>
        tensor& operator-= (
            const matrix_exp<EXP>& item
        );
        /*!
            requires
                - num_samples() == item.nr()
                - k()*nr()*nc() == item.nc()
                - item contains float values
            ensures
                - Subtracts item from *this tensor by performing:
                  set_ptrm(host(), num_samples(), k()*nr()*nc()) -= item;
        !*/

        template <typename EXP>
        void set_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        );
        /*!
            requires
                - idx < num_samples()
                - k()*nr()*nc() == item.size()
                - item contains float values
            ensures
                - Assigns item to the idx'th sample in *this by performing:
                  set_ptrm(host()+idx*item.size(), item.nr(), item.nc()) = item;
        !*/


        template <typename EXP>
        void add_to_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        );
        /*!
            requires
                - idx < num_samples()
                - k()*nr()*nc() == item.size()
                - item contains float values
            ensures
                - Adds item to the idx'th sample in *this by performing:
                  set_ptrm(host()+idx*item.size(), item.nr(), item.nc()) += item;
        !*/

    protected:

        // You can't move or copy another tensor into *this since that might modify the
        // tensor's dimensions.  If you want to do that sort of thing then use a
        // resizable_tensor.
        tensor(const tensor& item);  
        tensor& operator= (const tensor& item); 
        tensor(tensor&& item); 
        tensor& operator=(tensor&& item); 
    };

// ----------------------------------------------------------------------------------------

    void memcpy (
        tensor& dest, 
        const tensor& src
    );
    /*!
        requires
            - dest.size() == src.size()
        ensures
            - Copies the data in src to dest.  If the device data is current on both src
              and dest then the copy will happen entirely on the device side.
            - It doesn't matter what GPU device is selected by cudaSetDevice().  You can
              always copy tensor objects to and from each other regardless.
            - This function blocks until the copy has completed.
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp mat (
        const tensor& t,
        long nr,
        long nc
    );
    /*!
        requires
            - nr >= 0
            - nc >= 0
            - nr*nc == t.size()
        ensures
            - returns a matrix M such that:
                - M.nr() == nr
                - m.nc() == nc 
                - for all valid r and c:
                  M(r,c) == t.host()[r*nc + c]
                  (i.e. the tensor is interpreted as a matrix laid out in memory
                  in row major order)
    !*/

    const matrix_exp mat (
        const tensor& t
    );
    /*!
        ensures
            - if (t.size() != 0) then
                - returns mat(t, t.num_samples(), t.size()/t.num_samples())
            - else
                - returns an empty matrix.
    !*/

    const matrix_exp image_plane (
        const tensor& t,
        long sample = 0,
        long k = 0
    );
    /*!
        requires
            - t.size() != 0
            - 0 <= sample < t.num_samples()
            - 0 <= k < t.k()
        ensures
            - returns the k-th image plane from the sample-th image in t.  That is,
              returns a matrix M such that:
                - M contains float valued elements.
                - M.nr() == t.nr()
                - M.nc() == t.nc()
                - for all valid r and c:
                    - M(r,c) == t.host()[((sample*t.k() + k)*t.nr() + r)*t.nc() + c]
    !*/

// ----------------------------------------------------------------------------------------

    bool have_same_dimensions (
        const tensor& a,
        const tensor& b
    );
    /*!
        ensures
            - returns true if and only if all of the fallowing are satisfied:
                - a.num_samples() == b.num_samples() 
                - a.k()  == b.k() 
                - a.nr() == b.nr() 
                - a.nc() == b.nc()
    !*/

// ----------------------------------------------------------------------------------------

    class resizable_tensor : public tensor
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is just a tensor with the additional ability to be resized.
        !*/

    public:
        resizable_tensor(
        );
        /*!
            ensures
                - #size() == 0
                - #num_samples() == 0
                - #k() == 0
                - #nr() == 0
                - #nc() == 0
        !*/

        template <typename EXP>
        resizable_tensor(
            const matrix_exp<EXP>& item
        );
        /*!
            requires
                - item contains float values
            ensures
                - #num_samples() == item.nr()
                - #k() == item.nc()
                - #nr() == 1
                - #nc() == 1
                - Assigns item to *this tensor by performing:
                  set_ptrm(host(), num_samples(), k()*nr()*nc()) = item;
        !*/

        explicit resizable_tensor(
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        );
        /*!
            requires
                - n_ >= 0
                - k_ >= 0
                - nr_ >= 0
                - nc_ >= 0
            ensures
                - #size() == n_*k_*nr_*nc_
                - #num_samples() == n_
                - #k() == k_
                - #nr() == nr_
                - #nc() == nc_
        !*/

        // This object is copyable and movable
        resizable_tensor(const resizable_tensor&) = default;
        resizable_tensor(resizable_tensor&&) = default;
        resizable_tensor& operator= (const resizable_tensor&) = default;
        resizable_tensor& operator= (resizable_tensor&&) = default;

        void clear(
        );
        /*!
            ensures
                - #size() == 0
                - #num_samples() == 0
                - #k() == 0
                - #nr() == 0
                - #nc() == 0
                - #annotation().is_empty() == true
        !*/

        void copy_size (
            const tensor& item
        );
        /*!
            ensures
                - resizes *this so that: have_same_dimensions(#*this, item)==true
        !*/

        void set_size(
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        );
        /*!
            requires
                - n_ >= 0
                - k_ >= 0
                - nr_ >= 0
                - nc_ >= 0
            ensures
                - #size() == n_*k_*nr_*nc_
                - #num_samples() == n_
                - #k() == k_
                - #nr() == nr_
                - #nc() == nc_
        !*/

        template <typename EXP>
        resizable_tensor& operator= (
            const matrix_exp<EXP>& item
        );
        /*!
            requires
                - item contains float values
            ensures
                - if (num_samples() == item.nr() && k()*nr()*nc() == item.nc()) then
                    - the dimensions of this tensor are not changed
                - else
                    - #num_samples() == item.nr()
                    - #k() == item.nc()
                    - #nr() == 1
                    - #nc() == 1
                - Assigns item to *this tensor by performing:
                  set_ptrm(host(), num_samples(), k()*nr()*nc()) = item;
        !*/
    };

    void serialize(const tensor& item, std::ostream& out);
    void deserialize(resizable_tensor& item, std::istream& in);
    /*!
        provides serialization support for tensor and resizable_tensor.  Note that you can
        serialize to/from any combination of tenor and resizable_tensor objects.
    !*/

// ----------------------------------------------------------------------------------------

    double dot(
        const tensor& a,
        const tensor& b
    );
    /*!
        requires
            - a.size() == b.size()
        ensures
            - returns the dot product between a and b when they are both treated as
              a.size() dimensional vectors.  That is, this function pointwise multiplies
              the vectors together, then sums the result and returns it.

    !*/

// ----------------------------------------------------------------------------------------

    class alias_tensor_instance : public tensor
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tensor that aliases another tensor.  That is, it doesn't
                have its own block of memory but instead simply holds pointers to the
                memory of another tensor object.  It therefore allows you to efficiently
                break a tensor into pieces and pass those pieces into functions.

                An alias_tensor_instance doesn't own the resources it points to in any sense.
                So it is important to make sure that the underlying owning tensor doesn't get
                destructed before any alias tensors which point to it are destructed.
        !*/

        // You can't default initialize this object.  You can only get instances of it from
        // alias_tensor::operator().
        alias_tensor_instance(
        ); 
    };

    class alias_tensor_const_instance 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is essentially a const version of alias_tensor_instance and therefore
                represents a tensor.  However, due to the mechanics of C++, this object
                can't inherit from tensor.  So instead it provides a get() and an implicit
                conversion to const tensor.
        !*/

    public:

        // Methods that cast the alias to a tensor.
        const tensor& get() const;
        operator const tensor& (); 

    private:
        // You can't default initialize this object.  You can only get instances of it from
        // alias_tensor::operator().
        alias_tensor_const_instance();
    };

    class alias_tensor 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a tool for creating tensor objects that alias other tensor objects.
                That is, it allows you to make a tensor that references the memory space of
                another tensor object rather than owning its own memory.  This allows you
                to do things like interpret a single tensor in different ways or even as a
                group of multiple tensors.
        !*/
    public:

        alias_tensor (
        );
        /*!
            ensures
                - #size() == 0 
                - #num_samples() == 0
                - #k() == 0
                - #nr() == 0
                - #nc() == 0
        !*/

        alias_tensor (
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        );
        /*!
            requires
                - n_ >= 0
                - k_ >= 0
                - nr_ >= 0
                - nc_ >= 0
            ensures
                - #size() == n_*k_*nr_*nc_
                - #num_samples() == n_
                - #k() == k_
                - #nr() == nr_
                - #nc() == nc_
        !*/

        long num_samples() const; 
        long k() const; 
        long nr() const; 
        long nc() const; 
        size_t size() const;

        alias_tensor_instance operator() (
            tensor& t,
            size_t offset
        );
        /*!
            requires
                - offset+size() <= t.size()
            ensures
                - Returns a tensor that simply aliases the elements of t beginning with t's
                  offset'th element.  Specifically, this function returns an aliasing
                  tensor T such that:
                    - T.size()   == size()
                    - T.num_samples() == num_samples()
                    - T.k()      == k()
                    - T.nr()     == nr()
                    - T.nc()     == nc()
                    - T.host()   == t.host()+offset
                    - T.device() == t.device()+offset
                    - &T.annotation() == &t.annotation()
        !*/

        alias_tensor_const_instance operator() (
            const tensor& t,
            size_t offset
        );
        /*!
            requires
                - offset+size() <= t.size()
            ensures
                - This function is identical to the above version of operator() except that 
                  it takes and returns const tensors instead of non-const tensors.
        !*/
    };

    void serialize(const alias_tensor& item, std::ostream& out);
    void deserialize(alias_tensor& item, std::istream& in);
    /*!
        provides serialization support for alias_tensor.  
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TENSOR_ABSTRACT_H_


