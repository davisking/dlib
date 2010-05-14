// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SVM_C_EKm_TRAINER_ABSTRACT_H__
#ifdef DLIB_SVM_C_EKm_TRAINER_ABSTRACT_H__

#include "../algs.h"
#include "function_abstract.h"
#include "kernel_abstract.h"
#include "empirical_kernel_map_abstract.h"
#include "svm_c_linear_trainer_abstract.h"

namespace dlib
{
    template <
        typename K 
        >
    class svm_c_ekm_trainer
    {
        /*!
            REQUIREMENTS ON K 
                is a kernel function object as defined in dlib/svm/kernel_abstract.h 

            WHAT THIS OBJECT REPRESENTS
                This object represents a tool for training the C formulation of 
                a support vector machine.   It is implemented using the empirical_kernel_map
                to kernelize the svm_c_linear_trainer.  This makes it a very fast algorithm
                but means the user must supply a set of basis vectors.  

                For details about the "basis vectors" see the empirical_kernel_map 
                documentation.  In particular, see it's example program.
        !*/

    public:
        typedef K kernel_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef decision_function<kernel_type> trained_function_type;

        svm_c_ekm_trainer (
        );
        /*!
            ensures
                - This object is properly initialized and ready to be used
                  to train a support vector machine.
                - #get_oca() == oca() (i.e. an instance of oca with default parameters) 
                - #get_c_class1() == 1
                - #get_c_class2() == 1
                - #get_epsilon() == 0.001
                - #basis_loaded() == false
                - this object will not be verbose unless be_verbose() is called
        !*/

        explicit svm_c_ekm_trainer (
            const scalar_type& C 
        );
        /*!
            requires
                - C > 0
            ensures
                - This object is properly initialized and ready to be used
                  to train a support vector machine.
                - #get_oca() == oca() (i.e. an instance of oca with default parameters) 
                - #get_c_class1() == C
                - #get_c_class2() == C
                - #get_epsilon() == 0.001
                - #basis_loaded() == false
                - this object will not be verbose unless be_verbose() is called
        !*/

        void set_epsilon (
            scalar_type eps
        );
        /*!
            requires
                - eps > 0
            ensures
                - #get_epsilon() == eps 
        !*/

        const scalar_type get_epsilon (
        ) const;
        /*!
            ensures
                - returns the error epsilon that determines when training should stop.
                  Smaller values may result in a more accurate solution but take longer 
                  to execute.
        !*/

        void be_verbose (
        );
        /*!
            ensures
                - This object will print status messages to standard out so that a 
                  user can observe the progress of the algorithm.
        !*/

        void be_very_verbose (
        );
        /*!
            ensures
                - This object will print a lot of status messages to standard out so that a 
                  user can observe the progress of the algorithm.  In addition to the
                  few status messages normal verbosity produces this setting also causes
                  the underlying svm_c_linear_trainer to be verbose.
        !*/

        void be_quiet (
        );
        /*!
            ensures
                - this object will not print anything to standard out
        !*/

        void set_oca (
            const oca& item
        );
        /*!
            ensures
                - #get_oca() == item 
        !*/

        const oca get_oca (
        ) const;
        /*!
            ensures
                - returns a copy of the optimizer used to solve the SVM problem.  
        !*/

        const kernel_type get_kernel (
        ) const;
        /*!
            ensures
                - returns a copy of the kernel function in use by this object
        !*/

        void set_kernel (
            const kernel_type& k
        );
        /*!
            ensures
                - #get_kernel() == k 
        !*/

        template <typename T>
        void set_basis (
            const T& basis_samples
        );
        /*!
            requires
                - T must be a dlib::matrix type or something convertible to a matrix via vector_to_matrix()
                  (e.g. a std::vector)
                - is_vector(basis_samples) == true
                - basis_samples.size() > 0
                - get_kernel() must be capable of operating on the elements of basis_samples.  That is,
                  expressions such as get_kernel()(basis_samples(0), basis_samples(0)) should make sense.
            ensures
                - #basis_loaded() == true
        !*/

        bool basis_loaded (
        ) const;
        /*!
            ensures
                - returns true if this object has been loaded with basis vectors and false otherwise.
        !*/

        void set_c (
            scalar_type C 
        );
        /*!
            requires
                - C > 0
            ensures
                - #get_c_class1() == C 
                - #get_c_class2() == C 
        !*/

        const scalar_type get_c_class1 (
        ) const;
        /*!
            ensures
                - returns the SVM regularization parameter for the +1 class.  
                  It is the parameter that determines the trade off between
                  trying to fit the +1 training data exactly or allowing more errors 
                  but hopefully improving the generalization ability of the 
                  resulting classifier.  Larger values encourage exact fitting 
                  while smaller values of C may encourage better generalization. 
        !*/

        const scalar_type get_c_class2 (
        ) const;
        /*!
            ensures
                - returns the SVM regularization parameter for the -1 class.  
                  It is the parameter that determines the trade off between
                  trying to fit the -1 training data exactly or allowing more errors 
                  but hopefully improving the generalization ability of the 
                  resulting classifier.  Larger values encourage exact fitting 
                  while smaller values of C may encourage better generalization. 
        !*/

        void set_c_class1 (
            scalar_type C
        );
        /*!
            requires
                - C > 0
            ensures
                - #get_c_class1() == C
        !*/

        void set_c_class2 (
            scalar_type C
        );
        /*!
            requires
                - C > 0
            ensures
                - #get_c_class2() == C
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
            requires
                - basis_loaded() == true
                - is_binary_classification_problem(x,y) == true
                - x == a matrix or something convertible to a matrix via vector_to_matrix().
                  Also, x should contain sample_type objects.
                - y == a matrix or something convertible to a matrix via vector_to_matrix().
                  Also, y should contain scalar_type objects.
            ensures
                - trains a C support vector classifier given the training samples in x and 
                  labels in y.  
                - returns a decision function F with the following properties:
                    - if (new_x is a sample predicted have +1 label) then
                        - F(new_x) >= 0
                    - else
                        - F(new_x) < 0
        !*/

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y,
            scalar_type& svm_objective
        ) const;
        /*!
            requires
                - basis_loaded() == true
                - is_binary_classification_problem(x,y) == true
                - x == a matrix or something convertible to a matrix via vector_to_matrix().
                  Also, x should contain sample_type objects.
                - y == a matrix or something convertible to a matrix via vector_to_matrix().
                  Also, y should contain scalar_type objects.
            ensures
                - trains a C support vector classifier given the training samples in x and 
                  labels in y.  
                - #svm_objective == the final value of the SVM objective function
                - returns a decision function F with the following properties:
                    - if (new_x is a sample predicted have +1 label) then
                        - F(new_x) >= 0
                    - else
                        - F(new_x) < 0
        !*/

    }; 

}

#endif // DLIB_SVM_C_EKm_TRAINER_ABSTRACT_H__


