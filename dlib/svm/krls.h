// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KRLs_
#define DLIB_KRLs_

#include <vector>

#include "krls_abstract.h"
#include "../matrix.h"
#include "function.h"
#include "../std_allocator.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename kernel_type>
    class krls
    {
        /*!
            This is an implementation of the kernel recursive least squares algorithm described in the paper:
            The Kernel Recursive Least Squares Algorithm by Yaakov Engel.
        !*/

    public:
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;


        explicit krls (
            const kernel_type& kernel_, 
            scalar_type tolerance_ = 0.001
        ) : 
            kernel(kernel_), 
            tolerance(tolerance_)
        {
            clear_dictionary();
        }

        void set_tolerance (scalar_type tolerance_)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(tolerance_ >= 0,
                "\tvoid krls::set_tolerance"
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

        void clear_dictionary ()
        {
            dictionary.clear();
            alpha.clear();

            K_inv.set_size(0,0);
            P.set_size(0,0);
        }

        scalar_type operator() (
            const sample_type& x
        ) const
        {
            scalar_type temp = 0;
            for (unsigned long i = 0; i < alpha.size(); ++i)
                temp += alpha[i]*kernel(dictionary[i], x);

            return temp;
        }

        void train (
            const sample_type& x,
            scalar_type y
        )
        {
            const scalar_type kx = kern(x,x);
            if (alpha.size() == 0)
            {
                // set initial state since this is the first training example we have seen

                K_inv.set_size(1,1);
                K_inv(0,0) = 1/kx;

                alpha.push_back(y/kx);
                dictionary.push_back(x);
                P.set_size(1,1);
                P(0,0) = 1;
            }
            else
            {
                // fill in k
                k.set_size(alpha.size());
                for (long r = 0; r < k.nr(); ++r)
                    k(r) = kern(x,dictionary[r]);

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
                    // update the middle part of the matrix
                    set_subm(temp, get_rect(K_inv)) = K_inv + a*trans(a)/delta;
                    // update the right column of the matrix
                    set_subm(temp, 0, K_inv.nr(),K_inv.nr(),1) = -a/delta;
                    // update the bottom row of the matrix
                    set_subm(temp, K_inv.nr(), 0, 1, K_inv.nr()) = trans(-a/delta);
                    // update the bottom right corner of the matrix
                    temp(K_inv.nr(), K_inv.nc()) = 1/delta;
                    // put temp into K_inv
                    temp.swap(K_inv);


                    // Now update the P matrix (equation 3.15)
                    temp.set_size(P.nr()+1, P.nc()+1);
                    set_subm(temp, get_rect(P)) = P;
                    // initialize the new sides of P 
                    set_rowm(temp,P.nr()) = 0;
                    set_colm(temp,P.nr()) = 0;
                    temp(P.nr(), P.nc()) = 1;
                    temp.swap(P);

                    // now update the alpha vector (equation 3.16)
                    const scalar_type k_a = (y-trans(k)*vector_to_matrix(alpha))/delta;
                    for (unsigned long i = 0; i < alpha.size(); ++i)
                    {
                        alpha[i] -= a(i)*k_a;
                    }
                    alpha.push_back(k_a);
                }
                else
                {
                    q = P*a/(1+trans(a)*P*a);

                    // update P (equation 3.12)
                    temp_matrix = trans(a)*P;
                    P -= q*temp_matrix;

                    // update the alpha vector (equation 3.13)
                    const scalar_type k_a = y-trans(k)*vector_to_matrix(alpha);
                    for (unsigned long i = 0; i < alpha.size(); ++i)
                    {
                        alpha[i] += (K_inv*q*k_a)(i);
                    }
                }
            }
        }

        void swap (
            krls& item
        )
        {
            exchange(kernel, item.kernel);
            dictionary.swap(item.dictionary);
            alpha.swap(item.alpha);
            K_inv.swap(item.K_inv);
            P.swap(item.P);
            exchange(tolerance, item.tolerance);
            q.swap(item.q);
            a.swap(item.a);
            k.swap(item.k);
            temp_matrix.swap(item.temp_matrix);
        }

        unsigned long dictionary_size (
        ) const { return dictionary.size(); }

        decision_function<kernel_type> get_decision_function (
        ) const
        {
            return decision_function<kernel_type>(
                vector_to_matrix(alpha),
                0, // the KRLS algorithm doesn't have a bias term
                kernel,
                vector_to_matrix(dictionary)
            );
        }

        friend void serialize(const krls& item, std::ostream& out)
        {
            serialize(item.kernel, out);
            serialize(item.dictionary, out);
            serialize(item.alpha, out);
            serialize(item.K_inv, out);
            serialize(item.P, out);
            serialize(item.tolerance, out);
        }

        friend void deserialize(krls& item, std::istream& in)
        {
            deserialize(item.kernel, in);
            deserialize(item.dictionary, in);
            deserialize(item.alpha, in);
            deserialize(item.K_inv, in);
            deserialize(item.P, in);
            deserialize(item.tolerance, in);
        }

    private:

        inline scalar_type kern (const sample_type& m1, const sample_type& m2) const
        { 
            return kernel(m1,m2) + 0.001;
        }


        kernel_type kernel;

        typedef std_allocator<sample_type, mem_manager_type> alloc_sample_type;
        typedef std_allocator<scalar_type, mem_manager_type> alloc_scalar_type;
        typedef std::vector<sample_type,alloc_sample_type> dictionary_vector_type;
        typedef std::vector<scalar_type,alloc_scalar_type> alpha_vector_type;

        dictionary_vector_type dictionary;
        alpha_vector_type alpha;

        matrix<scalar_type,0,0,mem_manager_type> K_inv;
        matrix<scalar_type,0,0,mem_manager_type> P;

        scalar_type tolerance;


        // temp variables here just so we don't have to reconstruct them over and over.  Thus, 
        // they aren't really part of the state of this object.
        matrix<scalar_type,0,1,mem_manager_type> q;
        matrix<scalar_type,0,1,mem_manager_type> a;
        matrix<scalar_type,0,1,mem_manager_type> k;
        matrix<scalar_type,1,0,mem_manager_type> temp_matrix;

    };

// ----------------------------------------------------------------------------------------

    template <typename kernel_type>
    void swap(krls<kernel_type>& a, krls<kernel_type>& b)
    { a.swap(b); }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KRLs_

