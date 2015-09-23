// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_TENSOR_H_
#define DLIB_DNn_TENSOR_H_

#include <memory>
#include <cstring>
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class gpu_data 
    {
        /*!
            CONVENTION
                - if (size() != 0) then
                    - data_host == a pointer to size() floats in CPU memory.
                - if (data_device) then 
                    - data_device == a pointer to size() floats in device memory.


                - We use the host_current and device_current bools to keep track of which
                  copy of the data (or both) are most current.  e.g. if the CPU has
                  modified the tensor and it hasn't been copied to the device yet then
                  host_current==true and device_current == false.

        !*/
    public:

        gpu_data(
        ) : data_size(0), host_current(true), device_current(false)
        {
        }

        // Not copyable
        gpu_data(const gpu_data&) = delete;
        gpu_data& operator=(const gpu_data&) = delete;

        // but is movable
        gpu_data(gpu_data&&) = default;
        gpu_data& operator=(gpu_data&&) = default;

        void set_size(size_t new_size)
        {
            if (new_size == 0)
            {
                data_size = 0;
                host_current = true;
                device_current = false;
                data_host.reset();
                data_device.reset();
            }
            else if (new_size != data_size)
            {
                data_size = new_size;
                host_current = true;
                device_current = false;
                data_host.reset(new float[new_size]);
                data_device.reset();
            }
        }

        void async_copy_to_device() 
        {
            // TODO
        }

        void async_copy_to_host() 
        {
            // TODO
        }

        const float* host() const 
        { 
            copy_to_host();
            return data_host.get(); 
        }

        float* host() 
        {
            copy_to_host();
            device_current = false;
            return data_host.get(); 
        }

        const float* device() const 
        { 
            copy_to_device();
            return data_device.get(); 
        }

        float* device() 
        {
            copy_to_device();
            host_current = false;
            return data_device.get(); 
        }

        size_t size() const { return data_size; }

    private:

        void copy_to_device() const
        {
            if (!device_current)
            {
                // TODO, cudamemcpy()
                device_current = true;
            }
        }

        void copy_to_host() const
        {
            if (!host_current)
            {
                // TODO, cudamemcpy()
                host_current = true;
            }
        }

        size_t data_size;
        mutable bool host_current;
        mutable bool device_current;

        std::unique_ptr<float[]> data_host;
        std::unique_ptr<float[]> data_device;
    };

// ----------------------------------------------------------------------------------------

    class tensor
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:

        tensor (
        ) : 
            m_n(0), m_nr(0), m_nc(0), m_k(0)
        {
        }

        inline virtual ~tensor() = 0;

        long num_samples() const { return m_n; }
        long nr() const { return m_nr; }
        long nc() const { return m_nc; }
        long k() const { return m_k; }
        size_t size() const { return data.size(); }

        void async_copy_to_host() 
        {
            data.async_copy_to_host();
        }

        void async_copy_to_device() 
        {
            data.async_copy_to_device();
        }
        /*!
            ensures
                - begin asynchronously copying this tensor to the GPU.

                NOTE that the "get device pointer" routine in this class
                will have to do some kind of synchronization that ensures
                the copy is finished.
        !*/

        const float* host() const { return data.host(); }
        float*       host()       { return data.host(); }
        const float* device() const { return data.device(); }
        float*       device()       { return data.device(); }

        tensor& operator= (float val)
        {
            // TODO, do on the device if that's where the memory is living right now.
            auto d = data.host();
            for (size_t i = 0; i < data.size(); ++i)
                d[i] = val;
        }

        template <typename EXP>
        tensor& operator= (const matrix_exp<EXP>& item)
        {
            DLIB_CASSERT(num_samples() == item.nr() &&
                         nr()*nc()*k() == item.nc(),"");
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");

            set_ptrm(data.host(), m_n, m_nr*m_nc*m_k) = item;
            return *this;
        }

        template <typename EXP>
        tensor& operator+= (const matrix_exp<EXP>& item)
        {
            DLIB_CASSERT(num_samples() == item.nr() &&
                         nr()*nc()*k() == item.nc(),"");
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(data.host(), m_n, m_nr*m_nc*m_k) += item;
            return *this;
        }

        template <typename EXP>
        tensor& operator-= (const matrix_exp<EXP>& item)
        {
            DLIB_CASSERT(num_samples() == item.nr() &&
                         nr()*nc()*k() == item.nc(),"");
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(data.host(), m_n, m_nr*m_nc*m_k) -= item;
            return *this;
        }

        template <typename EXP>
        void set_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        )
        {
            DLIB_CASSERT(idx < num_samples(), "");
            DLIB_CASSERT(item.size() == nr()*nc()*k(), "");
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(data.host()+idx*item.size(), item.nr(), item.nc()) = item;
        }


        template <typename EXP>
        void add_to_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        )
        {
            DLIB_CASSERT(idx < num_samples(), "");
            DLIB_CASSERT(item.size() == nr()*nc()*k(), "");
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(data.host()+idx*item.size(), item.nr(), item.nc()) += item;
        }




    protected:

        tensor& operator= (const tensor& item) 
        {
            m_n  = item.m_n;
            m_nr = item.m_nr;
            m_nc = item.m_nc;
            m_k  = item.m_k;
            data.set_size(item.data.size());
            std::memcpy(data.host(), item.data.host(), data.size()*sizeof(float));
            return *this;
        }

        tensor(
            const tensor& item
        )  
        {
            *this = item;
        }

        tensor(tensor&& item) = default;
        tensor& operator=(tensor&& item) = default;


        long m_n;
        long m_nr;
        long m_nc;
        long m_k;
        gpu_data data;
    };

    tensor::~tensor()
    {
    }

// ----------------------------------------------------------------------------------------

    const matrix_op<op_pointer_to_mat<float> > mat (
        const tensor& t,
        long nr,
        long nc
    )
    {
        DLIB_ASSERT(nr > 0 && nc > 0 , 
                    "\tconst matrix_exp mat(tensor, nr, nc)"
                    << "\n\t nr and nc must be bigger than 0"
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
        );
        DLIB_ASSERT(nr*nc == t.size() , 
                    "\tconst matrix_exp mat(tensor, nr, nc)"
                    << "\n\t The sizes don't match up."
                    << "\n\t nr*nc:    " << nr*nc
                    << "\n\t t.size(): " << t.size()
        );
        typedef op_pointer_to_mat<float> op;
        return matrix_op<op>(op(t.host(),nr,nc));
    }

    const matrix_op<op_pointer_to_mat<float> > mat (
        const tensor& t
    )
    {
        DLIB_ASSERT(t.size() != 0, 
                    "\tconst matrix_exp mat(tensor)"
                    << "\n\t The tensor can't be empty."
        );

        return mat(t, t.num_samples(), t.size()/t.num_samples());
    }

// ----------------------------------------------------------------------------------------

    inline bool have_same_dimensions (
        const tensor& a,
        const tensor& b
    )
    {
        return a.num_samples() == b.num_samples() &&
               a.nr() == b.nr() &&
               a.nc() == b.nc() &&
               a.k()  == b.k();
    }

// ----------------------------------------------------------------------------------------

    class resizable_tensor : public tensor
    {
    public:
        resizable_tensor(
        )
        {}

        explicit resizable_tensor(
            long n_, long nr_ = 1, long nc_ = 1, long k_ = 1
        ) 
        {
            set_size(n_,nr_,nc_,k_);
        }

        resizable_tensor(const resizable_tensor&) = default;
        resizable_tensor(resizable_tensor&&) = default;

        void clear(
        )
        {
            set_size(0,0,0,0);
        }

        void copy_size (
            const tensor& item
        )
        /*!
            ensures
                - resizes *this so that: have_same_dimensions(#*this, item)==true
        !*/
        {
            set_size(item.num_samples(), item.nr(), item.nc(), item.k());
        }

        resizable_tensor& operator= (float val)
        {
            tensor::operator=(val);
            return *this;
        }

        template <typename EXP>
        resizable_tensor& operator= (const matrix_exp<EXP>& item)
        {
            tensor::operator=(item);
            return *this;
        }

        template <typename EXP>
        resizable_tensor& operator+= (const matrix_exp<EXP>& item)
        {
            tensor::operator+=(item);
            return *this;
        }

        template <typename EXP>
        resizable_tensor& operator-= (const matrix_exp<EXP>& item)
        {
            tensor::operator-=(item);
            return *this;
        }

        template <typename EXP>
        void set_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        )
        {
            tensor::set_sample(idx, item);
        }

        template <typename EXP>
        void add_to_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        )
        {
            tensor::add_to_sample(idx, item);
        }

        resizable_tensor& operator= (const resizable_tensor&) = default;
        resizable_tensor& operator= (resizable_tensor&&) = default;

        resizable_tensor& operator= (const tensor& x) 
        {
            tensor::operator=(x);
            return *this;
        }

        void set_size(
            long n_, long nr_ = 1, long nc_ = 1, long k_ = 1
        )
        {
            m_n = n_;
            m_nr = nr_;
            m_nc = nc_;
            m_k = k_;
            data.set_size(m_n*m_nr*m_nc*m_k);
        }
    };

// ----------------------------------------------------------------------------------------

    inline double dot(
        const tensor& a,
        const tensor& b
    )
    {
        DLIB_CASSERT(a.size() == b.size(), "");
        const float* da = a.host();
        const float* db = b.host();
        double sum = 0;
        for (size_t i = 0; i < a.size(); ++i)
            sum += da[i]*db[i];
        return sum;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TENSOR_H_

