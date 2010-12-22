// Copyright (C) 2007  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CALCULATE_RHO_ANd_B_H__
#define DLIB_CALCULATE_RHO_ANd_B_H__


#include <limits>

namespace dlib 
{

// ----------------------------------------------------------------------------------------

    template <
        typename scalar_vector_type,
        typename scalar_vector_type2,
        typename scalar_type
        >
    void calculate_rho_and_b(
        const scalar_vector_type2& y,
        const scalar_vector_type& alpha,
        const scalar_vector_type& df,
        scalar_type& rho, 
        scalar_type& b
    ) 
    /*!
        requires
            - is_col_vector(y) == true
            - is_col_vector(alpha) == true
            - is_col_vector(df) == true
            - df.size() == alpha.size() == y.size()
        ensures
            - calculates the rho and b values associated with an SVM problem.
              This function is basically an implementation detail of the svm_nu_trainer
              and svm_c_trainer.  

              rho and b are defined in the following paper:
                Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector 
                machines, 2001. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm
    !*/
    {
        using namespace std;
        long num_p_free = 0;
        long num_n_free = 0;
        scalar_type sum_p_free = 0;
        scalar_type sum_n_free = 0;

        scalar_type upper_bound_p = -numeric_limits<scalar_type>::infinity();
        scalar_type upper_bound_n = -numeric_limits<scalar_type>::infinity();
        scalar_type lower_bound_p = numeric_limits<scalar_type>::infinity();
        scalar_type lower_bound_n = numeric_limits<scalar_type>::infinity();

        for(long i = 0; i < alpha.nr(); ++i)
        {
            if(y(i) == 1)
            {
                if(alpha(i) == 1)
                {
                    if (df(i) > upper_bound_p)
                        upper_bound_p = df(i);
                }
                else if(alpha(i) == 0)
                {
                    if (df(i) < lower_bound_p)
                        lower_bound_p = df(i);
                }
                else
                {
                    ++num_p_free;
                    sum_p_free += df(i);
                }
            }
            else
            {
                if(alpha(i) == 1)
                {
                    if (df(i) > upper_bound_n)
                        upper_bound_n = df(i);
                }
                else if(alpha(i) == 0)
                {
                    if (df(i) < lower_bound_n)
                        lower_bound_n = df(i);
                }
                else
                {
                    ++num_n_free;
                    sum_n_free += df(i);
                }
            }
        }

        scalar_type r1,r2;
        if(num_p_free > 0)
            r1 = sum_p_free/num_p_free;
        else
            r1 = (upper_bound_p+lower_bound_p)/2;

        if(num_n_free > 0)
            r2 = sum_n_free/num_n_free;
        else
            r2 = (upper_bound_n+lower_bound_n)/2;

        rho = (r1+r2)/2;
        b = (r1-r2)/2/rho;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CALCULATE_RHO_ANd_B_H__


