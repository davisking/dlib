// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PEGASoS_ABSTRACT_
#ifdef DLIB_PEGASoS_ABSTRACT_

#include <cmath>
#include "../algs.h"
#include "function.h"
#include "kernel.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename kern_type
        >
    class svm_pegasos
    {
        /*!
            REQUIREMENTS ON kern_type
                is a kernel function object as defined in dlib/svm/kernel_abstract.h 

            WHAT THIS OBJECT REPRESENTS
                This object implements an online algorithm for training a support 
                vector machine for solving binary classification problems.  

                The implementation of the Pegasos algorithm used by this object is based
                on the following excellent paper:
                    Pegasos: Primal estimated sub-gradient solver for SVM (2007)
                    by Yoram Singer, Nathan Srebro 
                    In ICML 
        !*/

    public:
        typedef kern_type kernel_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef decision_function<kernel_type> trained_function_type;

        svm_pegasos (
        );
        /*!
            ensures
                - this object is properly initialized 
                - #get_lambda() == 0.0001
                - #get_tolerance() == 0.01
                - #get_train_count() == 0
        !*/

        svm_pegasos (
            const kernel_type& kernel_, 
            const scalar_type& lambda_,
            const scalar_type& tolerance_
        );
        /*!
            requires
                - lambda_ > 0
                - tolerance_ > 0
            ensures
                - this object is properly initialized 
                - #get_lambda() == lambda_ 
                - #get_tolerance() == tolerance_
                - #get_kernel() == kernel_
                - #get_train_count() == 0
        !*/

        void clear (
        );
        /*!
            ensures
                - #get_train_count() == 0
                - This object has the state is had just after it was constructed
                  (e.g. clears out any memory of previous calls to train())
        !*/

        void set_kernel (
            kernel_type k
        );
        /*!
            ensures
                - #get_kernel() == k
                - #get_train_count() == 0
                  (i.e. clears any memory of previous training)
        !*/

        void set_tolerance (
            double tol
        );
        /*!
            requires
                - tol > 0
            ensures
                - #get_tolerance() == tol
                - #get_train_count() == 0
                  (i.e. clears any memory of previous training)
        !*/

        void set_lambda (
            scalar_type lambda_
        );
        /*!
            requires
                - lambda_ > 0
            ensures
                - #get_lambda() == tol
                - #get_train_count() == 0
                  (i.e. clears any memory of previous training)
        !*/

        unsigned long get_train_count (
        ) const;
        /*!
            ensures
                - returns how many times this->train() has been called
                  since this object was constructed or last cleared.  
        !*/

        const scalar_type get_lambda (
        ) const;
        /*!
        !*/

        const scalar_type get_tolerance (
        ) const;
        /*!
        !*/

        const kernel_type get_kernel (
        ) const;
        /*!
        !*/

        scalar_type train (
            const sample_type& x,
            const scalar_type& y
        );
        /*!
            requires
                - y == 1 || y == -1
            ensures
                - trains this svm using the given sample x and label y
                - returns the current learning rate
        !*/

        scalar_type operator() (
            const sample_type& x
        ) const;
        /*!
        !*/

        const decision_function<kernel_type> get_decision_function (
        ) const;
        /*!
        !*/

        void swap (
            svm_pegasos& item
        );
        /*!
        !*/

    }; 

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type
        >
    class batch_trainer 
    {
    public:
        typedef typename trainer_type::kernel_type kernel_type;
        typedef typename trainer_type::scalar_type scalar_type;
        typedef typename trainer_type::sample_type sample_type;
        typedef typename trainer_type::mem_manager_type mem_manager_type;
        typedef typename trainer_type::trained_function_type trained_function_type;


        batch_trainer (
        );
        /*!
        !*/

        batch_trainer (
            const trainer_type& trainer_, 
            const scalar_type min_learning_rate_,
            bool verbose_
        );
        /*!
        !*/

        const kernel_type get_kernel (
        ) const;
        /*!
        !*/

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y
        ) const;
        /*!
        !*/

    }; 

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type
        >
    const batch_trainer<trainer_type> batch (
        const trainer_type& trainer,
        const typename trainer_type::scalar_type min_learning_rate = 0.1
    ) { return batch_trainer<trainer_type>(trainer, min_learning_rate, false); }
    /*!
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type
        >
    const batch_trainer<trainer_type> verbose_batch (
        const trainer_type& trainer,
        const typename trainer_type::scalar_type min_learning_rate = 0.1
    ) { return batch_trainer<trainer_type>(trainer, min_learning_rate, true); }
    /*!
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PEGASoS_ABSTRACT_


