// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_EMPIRICAL_KERNEl_MAP_H_
#define DLIB_EMPIRICAL_KERNEl_MAP_H_

#include "../matrix.h"
#include "empirical_kernel_map_abstract.h"
#include <vector>
#include "../algs.h"
#include "kernel_matrix.h"
#include "function.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename kernel_type, typename EXP>
    const decision_function<kernel_type> convert_to_decision_function (
        const projection_function<kernel_type>& project_funct,
        const matrix_exp<EXP>& vect
    ) 
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(project_funct.out_vector_size() > 0 && is_vector(vect) && 
                    project_funct.out_vector_size() == vect.size() && project_funct.weights.nc() == project_funct.basis_vectors.size(),
            "\t const decision_function convert_to_decision_function()"
            << "\n\t Invalid inputs to this function."
            << "\n\t project_funct.out_vector_size():    " << project_funct.out_vector_size() 
            << "\n\t project_funct.weights.nc():         " << project_funct.weights.nc() 
            << "\n\t project_funct.basis_vectors.size(): " << project_funct.basis_vectors.size() 
            << "\n\t is_vector(vect):                    " << is_vector(vect) 
            << "\n\t vect.size():                        " << vect.size() 
            );

        return decision_function<kernel_type>(trans(project_funct.weights)*vect, 
                                              0, 
                                              project_funct.kernel_function,
                                              project_funct.basis_vectors);
    }

// ----------------------------------------------------------------------------------------

    template <typename kern_type>
    class empirical_kernel_map
    {
    public:

        struct empirical_kernel_map_error : public error
        {
            empirical_kernel_map_error(const std::string& message): error(message) {}
        };

        typedef kern_type kernel_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;

        void clear (
        )
        {
            empirical_kernel_map().swap(*this);
        }

        void load(
            const kernel_type& kernel_,
            const std::vector<sample_type>& basis_samples
        )
        {
            load(kernel_, vector_to_matrix(basis_samples));
        }

        template <typename EXP>
        void load(
            const kernel_type& kernel_,
            const matrix_exp<EXP>& basis_samples
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(basis_samples.size() > 0 && is_vector(basis_samples),
                "\tvoid empirical_kernel_map::load(kernel,basis_samples)"
                << "\n\t You have to give a non-empty set of basis_samples and it must be a vector"
                << "\n\t basis_samples.size():     " << basis_samples.size() 
                << "\n\t is_vector(basis_samples): " << is_vector(basis_samples) 
                << "\n\t this: " << this
                );

            // clear out the weights before we begin.  This way if an exception throws
            // this object will already be in the right state.
            weights.set_size(0,0);
            kernel = kernel_;
            basis.clear();
            basis.reserve(basis_samples.size());

            // find out the value of the largest norm of the elements in basis_samples.
            const scalar_type max_norm = max(diag(kernel_matrix(kernel, basis_samples)));
            // we will consider anything less than or equal to this number to be 0
            const scalar_type eps = max_norm*std::numeric_limits<scalar_type>::epsilon();

            // Copy all the basis_samples into basis but make sure we don't copy any samples
            // that have length 0
            for (long i = 0; i < basis_samples.size(); ++i)
            {
                const scalar_type norm = kernel(basis_samples(i), basis_samples(i));
                if (norm > eps)
                {
                    basis.push_back(basis_samples(i));
                }
            }

            if (basis.size() == 0)
            {
                clear();
                throw empirical_kernel_map_error("All basis_samples given to empirical_kernel_map::load() were zero vectors");
            }

            matrix<scalar_type,0,0,mem_manager_type> K(kernel_matrix(kernel, basis)), U,W,V;

            if (svd2(false,true,K,U,W,V))
            {
                clear();
                throw empirical_kernel_map_error("While loading empirical_kernel_map with data, SVD failed to converge.");
            }


            // now count how many elements of W are non-zero
            const long num_not_zero = sum(W>eps);

            // Really, this should never happen.  But I'm checking for good measure.
            if (num_not_zero == 0)
            {
                clear();
                throw empirical_kernel_map_error("While loading empirical_kernel_map with data, SVD failed");
            }

            weights.set_size(num_not_zero, basis.size());

            // now fill the weights matrix with the output of the SVD
            long counter = 0;
            for (long i =0; i < W.size(); ++i)
            {
                double val = W(i);
                if (val > eps)
                {
                    val = std::sqrt(val);
                    set_rowm(weights,counter) = rowm(trans(V),i)/val;
                    ++counter;
                }
            }

        }

        const kernel_type get_kernel (
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(out_vector_size() > 0,
                "\tconst kernel_type empirical_kernel_map::get_kernel()"
                << "\n\t You have to load this object with a kernel before you can call this function"
                << "\n\t this: " << this
                );

            return kernel;
        }

        long out_vector_size (
        ) const
        {
            return weights.nr();
        }

        template <typename EXP>
        const decision_function<kernel_type> convert_to_decision_function (
            const matrix_exp<EXP>& vect
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(out_vector_size() != 0 && is_vector(vect) && out_vector_size() == vect.size(),
                "\t const decision_function empirical_kernel_map::convert_to_decision_function()"
                << "\n\t Invalid inputs to this function."
                << "\n\t out_vector_size(): " << out_vector_size() 
                << "\n\t is_vector(vect):   " << is_vector(vect) 
                << "\n\t vect.size():       " << vect.size() 
                << "\n\t this: " << this
                );

            return decision_function<kernel_type>(trans(weights)*vect, 0, kernel, vector_to_matrix(basis));
        }

        template <typename EXP>
        const distance_function<kernel_type> convert_to_distance_function (
            const matrix_exp<EXP>& vect
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(out_vector_size() != 0 && is_vector(vect) && out_vector_size() == vect.size(),
                "\t const distance_function empirical_kernel_map::convert_to_distance_function()"
                << "\n\t Invalid inputs to this function."
                << "\n\t out_vector_size(): " << out_vector_size() 
                << "\n\t is_vector(vect):   " << is_vector(vect) 
                << "\n\t vect.size():       " << vect.size() 
                << "\n\t this: " << this
                );

            return distance_function<kernel_type>(trans(weights)*vect, dot(vect,vect), kernel, vector_to_matrix(basis));
        }

        const projection_function<kernel_type> get_projection_function (
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(out_vector_size() != 0,
                "\tconst projection_function empirical_kernel_map::get_projection_function()"
                << "\n\t You have to load this object with data before you can call this function"
                << "\n\t this: " << this
                );

            return projection_function<kernel_type>(weights, kernel, vector_to_matrix(basis));
        }

        const matrix<scalar_type,0,1,mem_manager_type>& project (
            const sample_type& samp
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(out_vector_size() != 0,
                "\tconst matrix empirical_kernel_map::project()"
                << "\n\t You have to load this object with data before you can call this function"
                << "\n\t this: " << this
                );

            temp1 = kernel_matrix(kernel, basis, samp);
            temp2 = weights*temp1;
            return temp2;
        }

        void swap (
            empirical_kernel_map& item
        )
        {
            basis.swap(item.basis);
            weights.swap(item.weights);
            std::swap(kernel, item.kernel);

            temp1.swap(item.temp1);
            temp2.swap(item.temp2);
        }

        friend void serialize (
            const empirical_kernel_map& item,
            std::ostream& out
        )
        {
            serialize(item.basis, out);
            serialize(item.weights, out);
            serialize(item.kernel, out);
        }

        friend void deserialize (
            empirical_kernel_map& item,
            std::istream& in 
        )
        {
            deserialize(item.basis, in);
            deserialize(item.weights, in);
            deserialize(item.kernel, in);
        }

    private:

        std::vector<sample_type> basis;
        matrix<scalar_type,0,0,mem_manager_type> weights;
        kernel_type kernel;

        // These members don't contribute to the logical state of this object.  They are
        // just here so that they don't have to be reallocated every time the project() function
        // is called.
        mutable matrix<scalar_type,0,1,mem_manager_type> temp1, temp2;

    };

    template <typename kernel_type>
    void swap (
        empirical_kernel_map<kernel_type>& a,
        empirical_kernel_map<kernel_type>& b
    ) { a.swap(b); }
    
// ----------------------------------------------------------------------------------------

}

#endif // DLIB_EMPIRICAL_KERNEl_MAP_H_

