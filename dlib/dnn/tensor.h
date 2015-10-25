// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_TENSOR_H_
#define DLIB_DNn_TENSOR_H_

#include "tensor_abstract.h"
#include <cstring>
#include "../matrix.h"
#include "cudnn_dlibapi.h"
#include "gpu_data.h"

namespace dlib
{
// ----------------------------------------------------------------------------------------

    class tensor
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:

        tensor (
        ) : 
            m_n(0), m_k(0), m_nr(0), m_nc(0)
        {
        }

        inline virtual ~tensor() = 0;

        long num_samples() const { return m_n; }
        long k() const { return m_k; }
        long nr() const { return m_nr; }
        long nc() const { return m_nc; }
        size_t size() const { return data.size(); }

        void async_copy_to_device() const
        {
            data.async_copy_to_device();
        }

        const float* host() const { return data.host(); }
        float*       host()       { return data.host(); }
        const float* device() const { return data.device(); }
        float*       device()       { return data.device(); }

        tensor& operator= (float val)
        {
#ifdef DLIB_USE_CUDA
            // If you are using CUDA then presumably you will be mostly using tensor's on
            // the GPU.  So unless you seem to be actively working with the host side's
            // data then we do this initialization on the device side since this avoids a
            // host to device transfer that would likely immediately follow.
            if (data.device_ready())
            {
                cuda::set_tensor(*this, val);
                return *this;
            }
#endif
            auto d = data.host();
            for (size_t i = 0; i < data.size(); ++i)
                d[i] = val;
            return *this;
        }

        tensor& operator*= (float val)
        {
#ifdef DLIB_USE_CUDA
            cuda::scale_tensor(*this, val);
            return *this;
#else
            auto d = data.host();
            for (size_t i = 0; i < data.size(); ++i)
                d[i] *= val;
            return *this;
#endif
        }
        
        tensor& operator/= (float val)
        {
            *this *= 1.0/val;
            return *this;
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


#ifdef DLIB_USE_CUDA
        const cuda::tensor_descriptor& get_cudnn_tensor_descriptor (
        ) const { return cudnn_descriptor; }
#endif



    protected:

        tensor& operator= (const tensor& item) 
        {
            m_n  = item.m_n;
            m_k  = item.m_k;
            m_nr = item.m_nr;
            m_nc = item.m_nc;
            data.set_size(item.data.size());
            std::memcpy(data.host(), item.data.host(), data.size()*sizeof(float));
#ifdef DLIB_USE_CUDA
            cudnn_descriptor.set_size(m_n,m_k,m_nr,m_nc);
#endif
            return *this;
        }

        tensor(
            const tensor& item
        )  
        {
            *this = item;
        }

        tensor(tensor&& item) : tensor() { swap(item); }
        tensor& operator=(tensor&& item) { swap(item); return *this; }

        void swap(tensor& item)
        {
            std::swap(m_n,    item.m_n);
            std::swap(m_k,    item.m_k);
            std::swap(m_nr,   item.m_nr);
            std::swap(m_nc,   item.m_nc);
            std::swap(data, item.data);
#ifdef DLIB_USE_CUDA
            std::swap(cudnn_descriptor, item.cudnn_descriptor);
#endif
        }


        long m_n;
        long m_k;
        long m_nr;
        long m_nc;
        gpu_data data;
#ifdef DLIB_USE_CUDA
        cuda::tensor_descriptor cudnn_descriptor;
#endif 
    };

    tensor::~tensor()
    {
    }

// ----------------------------------------------------------------------------------------

    inline const matrix_op<op_pointer_to_mat<float> > mat (
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

    inline const matrix_op<op_pointer_to_mat<float> > mat (
        const tensor& t
    )
    {
        DLIB_ASSERT(t.size() != 0, 
                    "\tconst matrix_exp mat(tensor)"
                    << "\n\t The tensor can't be empty."
        );

        return mat(t, t.num_samples(), t.size()/t.num_samples());
    }

    inline const matrix_op<op_pointer_to_mat<float> > image_plane (
        const tensor& t,
        long sample = 0,
        long k = 0
    )
    {
        DLIB_ASSERT(0 <= sample && sample < t.num_samples() &&
                    0 <= k && k < t.k() &&
                    t.size() != 0, 
                    "\tconst matrix_exp image_plane(tensor,sample,k)"
                    << "\n\t Invalid arguments were given to this function."
                    << "\n\t sample: " << sample
                    << "\n\t k:      " << k 
                    << "\n\t t.num_samples(): " << t.num_samples() 
                    << "\n\t t.k():           " << t.k() 
                    << "\n\t t.size():        " << t.size() 
        );


        typedef op_pointer_to_mat<float> op;
        return matrix_op<op>(op(t.host() + ((sample*t.k() + k)*t.nr())*t.nc(), 
                                t.nr(), 
                                t.nc()));
    }

// ----------------------------------------------------------------------------------------

    inline bool have_same_dimensions (
        const tensor& a,
        const tensor& b
    )
    {
        return a.num_samples() == b.num_samples() &&
               a.k()  == b.k() &&
               a.nr() == b.nr() &&
               a.nc() == b.nc();
    }

// ----------------------------------------------------------------------------------------

    class resizable_tensor : public tensor
    {
    public:
        resizable_tensor(
        )
        {}

        explicit resizable_tensor(
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        ) 
        {
            set_size(n_,k_,nr_,nc_);
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
            set_size(item.num_samples(), item.k(), item.nr(), item.nc());
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
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        )
        {
            m_n = n_;
            m_k = k_;
            m_nr = nr_;
            m_nc = nc_;
            data.set_size(m_n*m_k*m_nr*m_nc);
#ifdef DLIB_USE_CUDA
            cudnn_descriptor.set_size(m_n,m_k,m_nr,m_nc);
#endif
        }
    };

    inline void serialize(const tensor& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.num_samples(), out);
        serialize(item.k(), out);
        serialize(item.nr(), out);
        serialize(item.nc(), out);
        auto data = item.host();
        for (size_t i = 0; i < item.size(); ++i)
            serialize(data[i], out);
    }

    inline void deserialize(resizable_tensor& item, std::istream& in)
    {
        int version;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::resizable_tensor.");

        long num_samples=0, k=0, nr=0, nc=0;
        deserialize(num_samples, in);
        deserialize(k, in);
        deserialize(nr, in);
        deserialize(nc, in);
        item.set_size(num_samples, k, nr, nc);
        auto data = item.host();
        for (size_t i = 0; i < item.size(); ++i)
            deserialize(data[i], in);
    }

// ----------------------------------------------------------------------------------------

    inline double dot(
        const tensor& a,
        const tensor& b
    )
    {
        // TODO, do on GPU?
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

