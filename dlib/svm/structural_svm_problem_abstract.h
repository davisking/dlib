// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_STRUCTURAL_SVM_PRObLEM_ABSTRACT_H__
#ifdef DLIB_STRUCTURAL_SVM_PRObLEM_ABSTRACT_H__

#include "../optimization/optimization_oca_abstract.h"
#include "sparse_vector_abstract.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type,
        typename feature_vector_type_ 
        >
    class structural_svm_problem : public oca_problem<matrix_type> 
    {
    public:
        /*!
            INITIAL VALUE
                - get_epsilon() == 0.001
                - get_max_cache_size() == 10
                - get_c() == 1
                - This object will not be verbose

            WHAT THIS OBJECT REPRESENTS

        !*/

        typedef typename matrix_type::type scalar_type;
        typedef feature_vector_type_ feature_vector_type;

        structural_svm_problem (
        );
        /*!
            ensures
                - this object is properly initialized
        !*/

        void set_epsilon (
            scalar_type eps_
        );
        /*!
            requires
                - eps_ > 0
            ensures
                - #get_epsilon() == eps_
        !*/

        const scalar_type get_epsilon (
        ) const;
        /*!
        !*/

        void set_max_cache_size (
            unsigned long max_size
        );
        /*!
            ensures
                - #get_max_cache_size() == max_size
        !*/

        unsigned long get_max_cache_size (
        ) const; 
        /*!
            ensures
                - Returns the number of joint feature vectors per training sample kept in 
                  the separation oracle cache.  This cache is used to avoid unnecessary 
                  calls to the separation oracle.  Note that a value of 0 means that 
                  caching is not used at all.  This is appropriate if the separation
                  oracle is cheap to evaluate. 
        !*/

        void be_verbose (
        );
        /*!
        !*/

        void be_quiet(
        );
        /*!
        !*/

        scalar_type get_c (
        ) const; 
        /*!
        !*/

        void set_c (
            scalar_type C_
        );
        /*!
            requires
                - C_ > 0
            ensures
                - #get_c() == C_
        !*/

    // --------------------------------
    //     User supplied routines
    // --------------------------------

        virtual long get_num_dimensions (
        ) const = 0;
        /*!
            ensures
                - returns the dimensionality of a joint feature vector
        !*/

        virtual long get_num_samples (
        ) const = 0;
        /*!
            ensures
                - returns the number of training samples in this problem. 
        !*/

        virtual void get_truth_joint_feature_vector (
            long idx,
            feature_vector_type& psi 
        ) const = 0;
        /*!
            requires
                - 0 <= idx < get_num_samples()
            ensures
                - #psi == PSI(x_idx, y_idx)
                  (i.e. the joint feature vector for sample idx and its true label.)
        !*/

        virtual void separation_oracle (
            const long idx,
            const matrix_type& current_solution,
            scalar_type& loss,
            feature_vector_type& psi
        ) const = 0;
        /*!
            requires
                - 0 <= idx < get_num_samples()
                - current_solution.size() == get_num_dimensions()
            ensures
                - runs the separation oracle on the idx-th sample.  We define this as follows: 
                    - let X           == the idx-th input sample.
                    - let PSI(X,y)    == the joint feature vector for input X and an arbitrary label y.
                    - let F(X,y)      == dot(current_solution,PSI(X,y)).  
                    - let LOSS(idx,y) == the loss incurred for predicting that the ith-th sample
                      has a label of y.  

                        Then the separation oracle finds a Y such that: 
                            Y = argmax over all y: LOSS(idx,y) + F(X,y) 
                            (i.e. It finds the label which maximizes the above expression.)

                        Finally, we can define the outputs of this function as:
                        - #loss == LOSS(idx,Y) 
                        - #psi == PSI(X,Y) 
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SVM_PRObLEM_ABSTRACT_H__


