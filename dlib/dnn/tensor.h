// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_TENSOR_H_
#define DLIB_DNn_TENSOR_H_

#include "tensor_abstract.h"
#include <cstring>
#include "../matrix.h"
#include "cudnn_dlibapi.h"
#include "gpu_data.h"
#include "../byte_orderer.h"
#include <memory>
#include "../any.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class tensor;
    namespace cuda
    {
        void set_tensor (
            tensor& t,
            float value
        );

        void scale_tensor (
            tensor& t,
            float value
        );
    }

// ----------------------------------------------------------------------------------------

    class tensor
    {
    public:

        tensor (
        ) : 
            m_n(0), m_k(0), m_nr(0), m_nc(0), m_size(0)
        {
        }

        virtual ~tensor() {}

        long num_samples() const { return m_n; }
        long k() const { return m_k; }
        long nr() const { return m_nr; }
        long nc() const { return m_nc; }
        size_t size() const { return m_size; }

        typedef float* iterator;
        typedef const float* const_iterator;
        iterator       begin()       { return host(); }
        const_iterator begin() const { return host(); }
        iterator       end()         { return host()+size(); }
        const_iterator end() const   { return host()+size(); }

        void async_copy_to_device() const
        {
            data().async_copy_to_device();
        }

        virtual const float* host() const = 0;
        virtual float*       host() = 0; 
        virtual float*       host_write_only() = 0;
        virtual const float* device() const = 0;
        virtual float*       device() = 0;
        virtual float*       device_write_only() = 0;

        virtual const any&   annotation() const = 0;
        virtual any&         annotation() = 0;

        int device_id() const { return data().device_id(); }

        tensor& operator= (float val)
        {
#ifdef DLIB_USE_CUDA
            // If you are using CUDA then presumably you will be mostly using tensors on
            // the GPU.  So unless you seem to be actively working with the host side's
            // data then we do this initialization on the device side since this avoids a
            // host to device transfer that would likely immediately follow.
            if (data().device_ready())
            {
                cuda::set_tensor(*this, val);
                return *this;
            }
#endif
            auto d = host_write_only();
            for (size_t i = 0; i < size(); ++i)
                d[i] = val;

            return *this;
        }

        tensor& operator*= (float val)
        {
#ifdef DLIB_USE_CUDA
            cuda::scale_tensor(*this, val);
            return *this;
#else
            for (auto& d : *this)
                d *= val;

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
                         nr()*nc()*k() == item.nc());
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");

            set_ptrm(host_write_only(), m_n, m_nr*m_nc*m_k) = item;
            return *this;
        }

        template <typename EXP>
        tensor& operator+= (const matrix_exp<EXP>& item)
        {
            DLIB_CASSERT(num_samples() == item.nr() &&
                         nr()*nc()*k() == item.nc());
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(host(), m_n, m_nr*m_nc*m_k) += item;
            return *this;
        }

        template <typename EXP>
        tensor& operator-= (const matrix_exp<EXP>& item)
        {
            DLIB_CASSERT(num_samples() == item.nr() &&
                         nr()*nc()*k() == item.nc());
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(host(), m_n, m_nr*m_nc*m_k) -= item;
            return *this;
        }

        template <typename EXP>
        void set_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        )
        {
            DLIB_CASSERT(idx < (unsigned long)num_samples());
            DLIB_CASSERT(item.size() == nr()*nc()*k());
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(host()+idx*item.size(), item.nr(), item.nc()) = item;
        }


        template <typename EXP>
        void add_to_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        )
        {
            DLIB_CASSERT(idx < (unsigned long)num_samples());
            DLIB_CASSERT(item.size() == nr()*nc()*k());
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(host()+idx*item.size(), item.nr(), item.nc()) += item;
        }


#ifdef DLIB_USE_CUDA
        virtual const cuda::tensor_descriptor& get_cudnn_tensor_descriptor (
        ) const = 0; 
#endif

        friend void memcpy (
            tensor& dest, 
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.size() == src.size());
            memcpy(dest.data(), dest.get_alias_offset(),  
                   src.data(),  src.get_alias_offset(), 
                   src.size());
        }


    protected:

        friend class alias_tensor;

        virtual gpu_data& data() = 0;
        virtual const gpu_data& data() const = 0;
        virtual size_t get_alias_offset() const { return 0; } // needed by alias_tensor.

        long m_n;
        long m_k;
        long m_nr;
        long m_nc;
        long m_size; // always equal to m_n*m_k*m_nr*m_nc
    };

// ----------------------------------------------------------------------------------------

    inline bool is_vector (
        const tensor& t
    )
    {
        return t.size() == (size_t)t.num_samples() ||
               t.size() == (size_t)t.k() ||
               t.size() == (size_t)t.nr() ||
               t.size() == (size_t)t.nc();
    }

// ----------------------------------------------------------------------------------------

    inline const matrix_op<op_pointer_to_mat<float> > mat (
        const tensor& t,
        long nr,
        long nc
    )
    {
        DLIB_ASSERT(nr >= 0 && nc >= 0 , 
                    "\tconst matrix_exp mat(tensor, nr, nc)"
                    << "\n\t nr and nc must be >= 0"
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
        );
        DLIB_ASSERT(nr*nc == (long)t.size() , 
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
        if (t.size() != 0)
            return mat(t, t.num_samples(), t.size()/t.num_samples());
        else
            return mat((float*)0,0,0);
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

        template <typename EXP>
        resizable_tensor(
            const matrix_exp<EXP>& item
        )
        {
            set_size(item.nr(), item.nc());
            *this = item;
        }

        explicit resizable_tensor(
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        ) 
        {
            DLIB_ASSERT( n_ >= 0 && k_ >= 0 && nr_ >= 0 && nc_ >= 0);

            set_size(n_,k_,nr_,nc_);
        }

        resizable_tensor(const resizable_tensor& item) : _annotation(item.annotation()) 
        {
            copy_size(item);
            memcpy(data_instance, item.data_instance);
        }
        resizable_tensor(const tensor& item) : _annotation(item.annotation()) 
        {
            copy_size(item);
            memcpy(*this, item);
        }

        resizable_tensor(resizable_tensor&& item) { swap(item); }
        resizable_tensor& operator=(resizable_tensor&& item) { swap(item); return *this; }

        virtual const float* host() const { return data_instance.host(); }
        virtual float*       host()       { return data_instance.host(); }
        virtual float*       host_write_only() { return data_instance.host_write_only(); }
        virtual const float* device() const { return data_instance.device(); }
        virtual float*       device()       { return data_instance.device(); }
        virtual float*       device_write_only() { return data_instance.device_write_only(); }

        virtual const any&   annotation() const { return _annotation; }
        virtual any&         annotation() { return _annotation; }

        void clear(
        )
        {
            set_size(0,0,0,0);
            _annotation.clear();
        }

        void copy_size (
            const tensor& item
        )
        {
            set_size(item.num_samples(), item.k(), item.nr(), item.nc());
        }

        resizable_tensor& operator= (float val)
        {
            tensor::operator=(val);
            return *this;
        }

        template <typename EXP>
        resizable_tensor& operator= (
            const matrix_exp<EXP>& item
        )
        {
            if (!(num_samples() == item.nr() && k()*nr()*nc() == item.nc()))
                set_size(item.nr(), item.nc());
            tensor::operator=(item);
            return *this;
        }

        void set_size(
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        )
        {
            DLIB_ASSERT( n_ >= 0 && k_ >= 0 && nr_ >= 0 && nc_ >= 0);

            m_n = n_;
            m_k = k_;
            m_nr = nr_;
            m_nc = nc_;
            m_size = n_*k_*nr_*nc_;
            data_instance.set_size(m_size);
#ifdef DLIB_USE_CUDA
            cudnn_descriptor.set_size(m_n,m_k,m_nr,m_nc);
#endif
        }


        resizable_tensor& operator= (const resizable_tensor& item) 
        {
            resizable_tensor temp(item);
            temp.swap(*this);
            return *this;
        }

        resizable_tensor& operator= (const tensor& item) 
        {
            resizable_tensor temp(item);
            temp.swap(*this);
            return *this;
        }


        void swap(resizable_tensor& item)
        {
            std::swap(m_n,    item.m_n);
            std::swap(m_k,    item.m_k);
            std::swap(m_nr,   item.m_nr);
            std::swap(m_nc,   item.m_nc);
            std::swap(m_size, item.m_size);
            std::swap(data_instance, item.data_instance);
            std::swap(_annotation, item._annotation);
#ifdef DLIB_USE_CUDA
            std::swap(cudnn_descriptor, item.cudnn_descriptor);
#endif
        }

#ifdef DLIB_USE_CUDA
        virtual const cuda::tensor_descriptor& get_cudnn_tensor_descriptor (
        ) const { return cudnn_descriptor; }
#endif

    private:

#ifdef DLIB_USE_CUDA
        cuda::tensor_descriptor cudnn_descriptor;
#endif 

        gpu_data data_instance;
        any _annotation;
        virtual gpu_data& data() { return data_instance; }
        virtual const gpu_data& data() const { return data_instance; }
    };

    inline void serialize(const tensor& item, std::ostream& out)
    {
        int version = 2;
        serialize(version, out);
        serialize(item.num_samples(), out);
        serialize(item.k(), out);
        serialize(item.nr(), out);
        serialize(item.nc(), out);
        byte_orderer bo;
        auto sbuf = out.rdbuf();
        for (auto d : item)
        {
            // Write out our data as 4byte little endian IEEE floats rather than using
            // dlib's default float serialization.  We do this because it will result in
            // more compact outputs.  It's slightly less portable but it seems doubtful
            // that any CUDA enabled platform isn't going to use IEEE floats.  But if one
            // does we can just update the serialization code here to handle it if such a
            // platform is encountered.
            bo.host_to_little(d);
            static_assert(sizeof(d)==4, "This serialization code assumes we are writing 4 byte floats");
            sbuf->sputn((char*)&d, sizeof(d));
        }
    }

    inline void deserialize(resizable_tensor& item, std::istream& in)
    {
        int version;
        deserialize(version, in);
        if (version != 2)
            throw serialization_error("Unexpected version found while deserializing dlib::resizable_tensor.");

        long num_samples=0, k=0, nr=0, nc=0;
        deserialize(num_samples, in);
        deserialize(k, in);
        deserialize(nr, in);
        deserialize(nc, in);
        item.set_size(num_samples, k, nr, nc);
        byte_orderer bo;
        auto sbuf = in.rdbuf();
        for (auto& d : item)
        {
            static_assert(sizeof(d)==4, "This serialization code assumes we are writing 4 byte floats");
            if (sbuf->sgetn((char*)&d,sizeof(d)) != sizeof(d))
            {
                in.setstate(std::ios::badbit);
                throw serialization_error("Error reading data while deserializing dlib::resizable_tensor.");
            }
            bo.little_to_host(d);
        }
    }

// ----------------------------------------------------------------------------------------

    inline double dot(
        const tensor& a,
        const tensor& b
    )
    {
        DLIB_CASSERT(a.size() == b.size());
        const float* da = a.host();
        const float* db = b.host();
        double sum = 0;
        for (size_t i = 0; i < a.size(); ++i)
            sum += da[i]*db[i];
        return sum;
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class alias_tensor_instance : public tensor
    {
        alias_tensor_instance(
        ) : data_instance(0), _annotation(0), data_offset(0) {}

    public:
        friend class alias_tensor;
        friend class alias_tensor_const_instance;

        alias_tensor_instance& operator= (float val)
        {
            tensor::operator=(val);
            return *this;
        }

        template <typename EXP>
        alias_tensor_instance& operator= (const matrix_exp<EXP>& item)
        {
            tensor::operator=(item);
            return *this;
        }

        virtual const float* host() const { return data_instance->host()+data_offset; }
        virtual float*       host()       { return data_instance->host()+data_offset; }
        virtual float*       host_write_only()    { return data_instance->host()+data_offset; }
        virtual const float* device() const { return data_instance->device()+data_offset; }
        virtual float*       device()       { return data_instance->device()+data_offset; }
        virtual float*       device_write_only()  { return data_instance->device()+data_offset; }

        virtual const any&   annotation() const { return *_annotation; }
        virtual any&         annotation() { return *_annotation; }

#ifdef DLIB_USE_CUDA
        virtual const cuda::tensor_descriptor& get_cudnn_tensor_descriptor (
        ) const { return *cudnn_descriptor; }
#endif
    private:

        virtual size_t get_alias_offset() const { return data_offset; } 

#ifdef DLIB_USE_CUDA
        std::shared_ptr<cuda::tensor_descriptor> cudnn_descriptor;
#endif
        gpu_data* data_instance;
        any* _annotation;
        size_t data_offset;
        virtual gpu_data& data() { return *data_instance; }
        virtual const gpu_data& data() const { return *data_instance; }
    };

// ----------------------------------------------------------------------------------------

    class alias_tensor_const_instance 
    {
    public:
        const tensor& get() const { return inst; }
        operator const tensor& () { return inst; }

        alias_tensor_const_instance(const alias_tensor_instance& item) : inst(item) {}

    private:
        alias_tensor_instance inst;

        friend class alias_tensor;
        alias_tensor_const_instance() {}
    };

// ----------------------------------------------------------------------------------------

    class alias_tensor 
    {
    public:

        alias_tensor (
        ) {}

        alias_tensor (
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        ) 
        {
            DLIB_ASSERT( n_ >= 0 && k_ >= 0 && nr_ >= 0 && nc_ >= 0);

            inst.m_n = n_;
            inst.m_k = k_;
            inst.m_nr = nr_;
            inst.m_nc = nc_;
            inst.m_size = n_*k_*nr_*nc_;
        }

        long num_samples(
        ) const { return inst.m_n; }

        long k(
        ) const { return inst.m_k; }

        long nr(
        ) const { return inst.m_nr; }

        long nc(
        ) const { return inst.m_nc; }

        size_t size(
        ) const { return inst.m_size; }

        alias_tensor_instance operator() (
            tensor& t,
            size_t offset
        ) const
        {
            DLIB_CASSERT(offset+size() <= t.size());

#ifdef DLIB_USE_CUDA
            if (!inst.cudnn_descriptor)
            {
                inst.cudnn_descriptor = std::make_shared<cuda::tensor_descriptor>();
                inst.cudnn_descriptor->set_size(inst.m_n, inst.m_k, inst.m_nr, inst.m_nc);
            }
#endif
            inst.data_instance = &t.data();
            inst._annotation   = &t.annotation();
            // Note that t might already be an aliasing tensor so we need to take that into
            // account.
            inst.data_offset = t.get_alias_offset()+offset;
            return inst;
        }

        alias_tensor_const_instance operator() (
            const tensor& t,
            size_t offset
        ) const
        {
            alias_tensor_const_instance temp;
            temp.inst = (*this)(const_cast<tensor&>(t),offset);
            return temp;
        }

    private:
        mutable alias_tensor_instance inst;
    };

    inline void serialize(const alias_tensor& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.num_samples(), out);
        serialize(item.k(), out);
        serialize(item.nr(), out);
        serialize(item.nc(), out);
    }

    inline void deserialize(alias_tensor& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::alias_tensor.");
        long num_samples, k, nr, nc;
        deserialize(num_samples, in);
        deserialize(k, in);
        deserialize(nr, in);
        deserialize(nc, in);
        item = alias_tensor(num_samples, k, nr, nc);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TENSOR_H_

