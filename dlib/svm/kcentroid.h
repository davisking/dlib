// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KCENTROId_
#define DLIB_KCENTROId_

#include <vector>

#include "kcentroid_abstract.h"
#include "../matrix.h"
#include "function.h"
#include "../std_allocator.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename kernel_type>
    class kcentroid
    {
        /*!
            This is an implementation of an online algorithm for recursively estimating the
            centroid of a sequence of training points.  It uses the sparsification technique
            described in the paper The Kernel Recursive Least Squares Algorithm by Yaakov Engel.

            To understand the code it would also be useful to consult page 114 of the book Kernel 
            Methods for Pattern Analysis by Taylor and Cristianini as well as page 554 
            (particularly equation 18.31) of the book Learning with Kernels by Scholkopf and Smola.
        !*/

    public:
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;


        explicit kcentroid (
            const kernel_type& kernel_, 
            scalar_type tolerance_ = 0.001
        ) : 
            kernel(kernel_), 
            tolerance(tolerance_),
            max_dis(1e6)
        {
            clear_dictionary();
        }

        void set_tolerance (scalar_type tolerance_)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(tolerance_ >= 0,
                "\tvoid kcentroid::set_tolerance"
                << "\n\tinvalid tolerance value"
                << "\n\ttolerance: " << tolerance_
                << "\n\tthis: " << this
                );
            tolerance = tolerance_;
        }

        scalar_type get_tolerance() const
        {
            return tolerance;
        }

        void set_max_discount (
            scalar_type value 
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(value >= 0,
                "\tvoid kcentroid::set_max_discount"
                << "\n\tinvalid discount value"
                << "\n\tvalue: " << value 
                << "\n\tthis: " << this
                );
            max_dis = value;
            if (samples_seen > value)
                samples_seen = value;
        }

        scalar_type get_max_discount(
        ) const
        {
            return max_dis;
        }

        void clear_dictionary ()
        {
            dictionary.clear();
            alpha.clear();

            K_inv.set_size(0,0);
            samples_seen = 0;
            bias = 0;
        }

        scalar_type operator() (
            const sample_type& x
        ) const
        {
            scalar_type temp = 0;
            for (unsigned long i = 0; i < alpha.size(); ++i)
                temp += alpha[i]*kernel(dictionary[i], x);

            return std::sqrt(kernel(x,x) + bias - 2*temp);
        }

        scalar_type test_and_train (
            const sample_type& x
        )
        {
            return train_and_maybe_test(x,true);
        }

        void train (
            const sample_type& x
        )
        {
            train_and_maybe_test(x,false);
        }

        void swap (
            kcentroid& item
        )
        {
            exchange(kernel, item.kernel);
            dictionary.swap(item.dictionary);
            alpha.swap(item.alpha);
            K_inv.swap(item.K_inv);
            K.swap(item.K);
            exchange(tolerance, item.tolerance);
            exchange(samples_seen, item.samples_seen);
            exchange(bias, item.bias);
            exchange(max_dis, item.max_dis);
            a.swap(item.a);
            k.swap(item.k);
        }

        unsigned long dictionary_size (
        ) const { return dictionary.size(); }

        friend void serialize(const kcentroid& item, std::ostream& out)
        {
            serialize(item.kernel, out);
            serialize(item.dictionary, out);
            serialize(item.alpha, out);
            serialize(item.K_inv, out);
            serialize(item.K, out);
            serialize(item.tolerance, out);
            serialize(item.samples_seen, out);
            serialize(item.bias, out);
            serialize(item.max_dis, out);
        }

        friend void deserialize(kcentroid& item, std::istream& in)
        {
            deserialize(item.kernel, in);
            deserialize(item.dictionary, in);
            deserialize(item.alpha, in);
            deserialize(item.K_inv, in);
            deserialize(item.K, in);
            deserialize(item.tolerance, in);
            deserialize(item.samples_seen, in);
            deserialize(item.bias, in);
            deserialize(item.max_dis, in);
        }

    private:

        scalar_type train_and_maybe_test (
            const sample_type& x,
            bool do_test
        )
        {
            scalar_type test_result = 0;
            const scalar_type kx = kernel(x,x);
            if (alpha.size() == 0)
            {
                // set initial state since this is the first training example we have seen

                K_inv.set_size(1,1);
                K_inv(0,0) = 1/kx;
                K.set_size(1,1);
                K(0,0) = kx;

                alpha.push_back(1.0);
                dictionary.push_back(x);
            }
            else
            {
                // fill in k
                k.set_size(alpha.size());
                for (long r = 0; r < k.nr(); ++r)
                    k(r) = kernel(x,dictionary[r]);

                if (do_test)
                {
                    test_result = std::sqrt(kx + bias - 2*trans(vector_to_matrix(alpha))*k);
                }

                // compute the error we would have if we approximated the new x sample
                // with the dictionary.  That is, do the ALD test from the KRLS paper.
                a = K_inv*k;
                const scalar_type delta = kx - trans(k)*a;

                // if this new vector isn't approximately linearly dependent on the vectors
                // in our dictionary.
                if (std::abs(delta) > tolerance)
                {
                    // add x to the dictionary
                    dictionary.push_back(x);

                    // update K_inv by computing the new one in the temp matrix (equation 3.14)
                    matrix<scalar_type,0,0,mem_manager_type> temp(K_inv.nr()+1, K_inv.nc()+1);
                    for (long r = 0; r < K_inv.nr(); ++r)
                    {
                        for (long c = 0; c < K_inv.nc(); ++c)
                        {
                            temp(r,c) = (K_inv + a*trans(a)/delta)(r,c);
                        }
                    }
                    temp(K_inv.nr(), K_inv.nc()) = 1/delta;

                    // update the new sides of K_inv
                    for (long i = 0; i < K_inv.nr(); ++i)
                    {
                        temp(K_inv.nr(),i) = -a(i)/delta;
                        temp(i,K_inv.nr()) = -a(i)/delta;
                    }
                    // put temp into K_inv
                    temp.swap(K_inv);



                    // update K (the kernel matrix)
                    temp.set_size(K.nr()+1, K.nc()+1);
                    for (long r = 0; r < K.nr(); ++r)
                    {
                        for (long c = 0; c < K.nc(); ++c)
                        {
                            temp(r,c) = K(r,c);
                        }
                    }
                    temp(K.nr(), K.nc()) = kx;

                    // update the new sides of K
                    for (long i = 0; i < K.nr(); ++i)
                    {
                        temp(K.nr(),i) = k(i);
                        temp(i,K.nr()) = k(i);
                    }
                    // put temp into K
                    temp.swap(K);




                    // now update the alpha vector 
                    const double alpha_scale = samples_seen/(samples_seen+1);
                    for (unsigned long i = 0; i < alpha.size(); ++i)
                    {
                        alpha[i] *= alpha_scale;
                    }
                    alpha.push_back(1.0-alpha_scale);
                }
                else
                {
                    // update the alpha vector so that this new sample has been added into
                    // the mean vector we are accumulating
                    const double alpha_scale = samples_seen/(samples_seen+1);
                    const double a_scale = 1.0-alpha_scale;
                    for (unsigned long i = 0; i < alpha.size(); ++i)
                    {
                        alpha[i] = alpha_scale*alpha[i] + a_scale*a(i);
                    }
                }
            }

            ++samples_seen;

            // recompute the bias term
            bias = sum(pointwise_multiply(K, vector_to_matrix(alpha)*trans(vector_to_matrix(alpha))));
            
            if (samples_seen > max_dis)
                samples_seen = max_dis;

            return test_result;
        }


        typedef std_allocator<sample_type, mem_manager_type> alloc_sample_type;
        typedef std_allocator<scalar_type, mem_manager_type> alloc_scalar_type;
        typedef std::vector<sample_type,alloc_sample_type> dictionary_vector_type;
        typedef std::vector<scalar_type,alloc_scalar_type> alpha_vector_type;


        kernel_type kernel;
        dictionary_vector_type dictionary;
        alpha_vector_type alpha;

        matrix<scalar_type,0,0,mem_manager_type> K_inv;
        matrix<scalar_type,0,0,mem_manager_type> K;

        scalar_type tolerance;
        scalar_type samples_seen;
        scalar_type bias;
        scalar_type max_dis;


        // temp variables here just so we don't have to reconstruct them over and over.  Thus, 
        // they aren't really part of the state of this object.
        matrix<scalar_type,0,1,mem_manager_type> a;
        matrix<scalar_type,0,1,mem_manager_type> k;

    };

// ----------------------------------------------------------------------------------------

    template <typename kernel_type>
    void swap(kcentroid<kernel_type>& a, kcentroid<kernel_type>& b)
    { a.swap(b); }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KCENTROId_

