// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DPCA_h_
#define DLIB_DPCA_h_

#include "dpca_abstract.h"
#include <limits>
#include <cmath>
#include "../algs.h"
#include "../matrix.h"
#include <iostream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename column_matrix
        >
    class discriminant_pca
    {
        /*!
            INITIAL VALUE
                - vect_size == 0
                - total_count == 0
                - between_count == 0
                - within_count == 0
                - between_weight == 1
                - within_weight == 1

            CONVENTION
                - vect_size == in_vector_size()
                - total_count == the number of times add_to_total_variance() has been called.
                - within_count == the number of times add_to_within_class_variance() has been called.
                - between_count == the number of times add_to_between_class_variance() has been called.
                - between_weight == between_class_weight()
                - within_weight == within_class_weight()

                - if (total_count != 0)
                    - total_sum == the sum of all vectors given to add_to_total_variance()
                    - the covariance of all the elements given to add_to_total_variance() is given
                      by:
                        - let avg == total_sum/total_count
                        - covariance == total_cov/total_count - avg*trans(avg)
                - if (within_count != 0)
                    - within_cov/within_count == the normalized within class scatter matrix  
                - if (between_count != 0)
                    - between_cov/between_count == the normalized between class scatter matrix  
        !*/

    public:

        struct discriminant_pca_error : public error
        {
            discriminant_pca_error(const std::string& message): error(message) {}
        };

        typedef typename column_matrix::mem_manager_type mem_manager_type;
        typedef typename column_matrix::type scalar_type;
        typedef typename column_matrix::layout_type layout_type;
        const static long N = column_matrix::NR;
        typedef matrix<scalar_type,0,0,mem_manager_type,layout_type> general_matrix;

        discriminant_pca (
        ) 
        {
            clear();
        }

        void clear(
        )
        {
            total_count = 0;
            between_count = 0;
            within_count = 0;

            vect_size = 0;


            between_weight = 1;
            within_weight = 1;


            total_sum.set_size(0,0);
            between_cov.set_size(0,0);
            total_cov.set_size(0,0);
            within_cov.set_size(0,0);
        }

        long in_vector_size (
        ) const
        {
            return vect_size;
        }

        void set_within_class_weight (
            scalar_type weight
        )
        {
            within_weight = weight;
        }

        scalar_type within_class_weight (
        ) const
        {
            return within_weight;
        }

        void set_between_class_weight (
            scalar_type weight
        )
        {
            between_weight = weight;
        }

        scalar_type between_class_weight (
        ) const
        {
            return between_weight;
        }

        void add_to_within_class_variance(
            const column_matrix& x,
            const column_matrix& y
        )
        {
            vect_size = x.size();
            if (within_count == 0)
            {
                within_cov = (x-y)*trans(x-y);
            }
            else
            {
                within_cov += (x-y)*trans(x-y);
            }
            ++within_count;
        }

        void add_to_between_class_variance(
            const column_matrix& x,
            const column_matrix& y
        )
        {
            vect_size = x.size();
            if (between_count == 0)
            {
                between_cov = (x-y)*trans(x-y);
            }
            else
            {
                between_cov += (x-y)*trans(x-y);
            }
            ++between_count;
        }

        void add_to_total_variance(
            const column_matrix& x
        )
        {
            vect_size = x.size();
            if (total_count == 0)
            {
                total_cov = x*trans(x);
                total_sum = x;
            }
            else
            {
                total_cov += x*trans(x);
                total_sum += x;
            }
            ++total_count;
        }

        const general_matrix dpca_matrix (
            const double eps = 0.99
        ) const
        {
            general_matrix dpca_mat;
            general_matrix eigenvalues;
            dpca_matrix(dpca_mat, eigenvalues, eps);
            return dpca_mat;
        }

        void dpca_matrix (
            general_matrix& dpca_mat,
            general_matrix& eigenvalues,
            const double eps = 0.99
        ) const
        {
            general_matrix cov;

            // now combine the three measures of variance into a single matrix by using the
            // within_weight and between_weight weights.
            cov = get_total_covariance_matrix();
            if (within_count != 0)
                cov -= within_weight*within_cov/within_count; 
            if (between_count != 0)
                cov += between_weight*between_cov/between_count; 


            eigenvalue_decomposition<general_matrix> eig(cov);

            eigenvalues = eig.get_real_eigenvalues();
            dpca_mat = eig.get_pseudo_v();

            // sort the eigenvalues and eigenvectors so that the biggest eigenvalues come first
            rsort_columns(dpca_mat, eigenvalues);

            // Some of the eigenvalues might be negative.  So first lets zero those out
            // so they won't get considered.
            eigenvalues = pointwise_multiply(eigenvalues > 0, eigenvalues);
            // figure out how many eigenvectors we want in our dpca matrix
            const double thresh = sum(eigenvalues)*eps;
            long num_vectors = 0;
            double total = 0;
            for (long r = 0; r < eigenvalues.size() && total < thresh; ++r)
            {
                // Don't even think about looking at eigenvalues that are 0.  If we go this
                // far then we have all we need.
                if (eigenvalues(r) == 0)
                    break;

                ++num_vectors;
                total += eigenvalues(r);
            }
            
            if (num_vectors == 0)
                throw discriminant_pca_error("While performing discriminant_pca, all eigenvalues were negative or 0");

            // So now we know we want to use num_vectors of the first eigenvectors.  So
            // pull those out and discard the rest.
            dpca_mat = trans(colm(dpca_mat,range(0,num_vectors-1)));

            // also clip off the eigenvalues we aren't using
            eigenvalues = rowm(eigenvalues, range(0,num_vectors-1));

        }

        void swap (
            discriminant_pca& item
        )
        {
            using std::swap;
            swap(total_cov, item.total_cov);
            swap(total_sum, item.total_sum);
            swap(total_count, item.total_count);
            swap(vect_size, item.vect_size);
            swap(between_cov, item.between_cov);

            swap(between_count, item.between_count);
            swap(between_weight, item.between_weight);
            swap(within_cov, item.within_cov);
            swap(within_count, item.within_count);
            swap(between_weight, item.between_weight);
        }

        friend void deserialize (
            discriminant_pca& item, 
            std::istream& in
        )
        {
            deserialize( item.total_cov, in);
            deserialize( item.total_sum, in);
            deserialize( item.total_count, in);
            deserialize( item.vect_size, in);
            deserialize( item.between_cov, in);
            deserialize( item.between_count, in);
            deserialize( item.between_weight, in);
            deserialize( item.within_cov, in);
            deserialize( item.within_count, in);
            deserialize( item.between_weight, in);
        }

        friend void serialize (
            const discriminant_pca& item, 
            std::ostream& out 
        )   
        {
            serialize( item.total_cov, out);
            serialize( item.total_sum, out);
            serialize( item.total_count, out);
            serialize( item.vect_size, out);
            serialize( item.between_cov, out);
            serialize( item.between_count, out);
            serialize( item.between_weight, out);
            serialize( item.within_cov, out);
            serialize( item.within_count, out);
            serialize( item.between_weight, out);
        }

    private:

        general_matrix get_total_covariance_matrix (
        ) const
        /*!
            ensures
                - returns the covariance matrix of all the data given to the add_to_total_variance()
        !*/
        {
            // if we don't even know the dimensionality of the vectors we are dealing
            // with then just return an empty matrix
            if (vect_size == 0)
                return general_matrix();

            // we know the vector size but we have zero total covariance.  
            if (total_count == 0)
            {
                general_matrix temp(vect_size,vect_size);
                temp = 0;
                return temp;
            }

            // In this case we actually have something to make a total covariance matrix out of. 
            // So do that.
            column_matrix avg = total_sum/total_count;

            return total_cov/total_count - avg*trans(avg);
        }

        general_matrix total_cov;
        general_matrix total_sum;
        long total_count;

        long vect_size;

        general_matrix between_cov;
        long between_count;
        scalar_type between_weight;

        general_matrix within_cov;
        long within_count;
        scalar_type within_weight;
    };

    template <
        typename column_matrix
        >
    inline void swap (
        discriminant_pca<column_matrix>& a, 
        discriminant_pca<column_matrix>& b 
    ) { a.swap(b); }   

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DPCA_h_


