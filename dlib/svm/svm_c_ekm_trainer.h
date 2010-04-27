// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SVM_C_EKm_TRAINER_H__
#define DLIB_SVM_C_EKm_TRAINER_H__

#include "../algs.h"
#include "function.h"
#include "kernel.h"
#include "empirical_kernel_map.h"
#include "svm_c_linear_trainer.h"
#include "svm_c_ekm_trainer_abstract.h"
#include "../statistics.h"
#include <vector>

namespace dlib
{
    template <
        typename K 
        >
    class svm_c_ekm_trainer
    {

    public:
        typedef K kernel_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef decision_function<kernel_type> trained_function_type;

        svm_c_ekm_trainer (
        )
        {
            verbose = false;
            ekm_stale = true;
        }

        explicit svm_c_ekm_trainer (
            const scalar_type& C 
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(C > 0,
                "\t svm_c_ekm_trainer::svm_c_ekm_trainer()"
                << "\n\t C must be greater than 0"
                << "\n\t C:    " << C 
                << "\n\t this: " << this
                );


            ocas.set_c(C);
            verbose = false;
            ekm_stale = true;
        }

        void set_epsilon (
            scalar_type eps
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(eps > 0,
                "\t void svm_c_ekm_trainer::set_epsilon()"
                << "\n\t eps must be greater than 0"
                << "\n\t eps: " << eps 
                << "\n\t this: " << this
                );

            ocas.set_epsilon(eps);
        }

        const scalar_type get_epsilon (
        ) const
        {
            return ocas.get_epsilon();
        }

        void be_verbose (
        ) 
        { 
            verbose = true;
            ocas.be_quiet(); 
        }

        void be_very_verbose (
        )
        {
            verbose = true;
            ocas.be_verbose(); 
        }

        void be_quiet (
        )
        { 
            verbose = false;
            ocas.be_quiet(); 
        }

        void set_oca (
            const oca& item
        )
        {
            ocas.set_oca(item);
        }

        const oca get_oca (
        ) const
        {
            return ocas.get_oca();
        }

        const kernel_type get_kernel (
        ) const
        {
            return kern;
        }

        void set_kernel (
            const kernel_type& k
        )
        {
            kern = k;
            ekm_stale = true;
        }

        template <typename T>
        void set_basis (
            const T& basis_samples
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(basis_samples.size() > 0 && is_vector(vector_to_matrix(basis_samples)),
                "\tvoid svm_c_ekm_trainer::set_basis(basis_samples)"
                << "\n\t You have to give a non-empty set of basis_samples and it must be a vector"
                << "\n\t basis_samples.size():                       " << basis_samples.size() 
                << "\n\t is_vector(vector_to_matrix(basis_samples)): " << is_vector(vector_to_matrix(basis_samples)) 
                << "\n\t this: " << this
                );

            basis = vector_to_matrix(basis_samples);
            ekm_stale = true;
        }

        bool basis_loaded(
        ) const
        {
            return (basis.size() != 0);
        }

        void set_c (
            scalar_type C 
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(C > 0,
                "\t void svm_c_ekm_trainer::set_c()"
                << "\n\t C must be greater than 0"
                << "\n\t C:    " << C 
                << "\n\t this: " << this
                );

            ocas.set_c(C);
        }

        const scalar_type get_c_class1 (
        ) const
        {
            return ocas.get_c_class1();
        }

        const scalar_type get_c_class2 (
        ) const
        {
            return ocas.get_c_class2();
        }

        void set_c_class1 (
            scalar_type C
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(C > 0,
                "\t void svm_c_ekm_trainer::set_c_class1()"
                << "\n\t C must be greater than 0"
                << "\n\t C:    " << C 
                << "\n\t this: " << this
                );

            ocas.set_c_class1(C);
        }

        void set_c_class2 (
            scalar_type C
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(C > 0,
                "\t void svm_c_ekm_trainer::set_c_class2()"
                << "\n\t C must be greater than 0"
                << "\n\t C:    " << C 
                << "\n\t this: " << this
                );

            ocas.set_c_class2(C);
        }

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y
        ) const
        {
            scalar_type obj;
            return do_train(vector_to_matrix(x),vector_to_matrix(y),obj);
        }

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y,
            scalar_type& svm_objective
        ) const
        {
            return do_train(vector_to_matrix(x),vector_to_matrix(y),svm_objective);
        }


    private:

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> do_train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y,
            scalar_type& svm_objective
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(basis_loaded() == true && is_binary_classification_problem(x,y) == true,
                "\t decision_function svm_c_ekm_trainer::train(x,y)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t x.nr(): " << x.nr() 
                << "\n\t y.nr(): " << y.nr() 
                << "\n\t x.nc(): " << x.nc() 
                << "\n\t y.nc(): " << y.nc() 
                << "\n\t is_binary_classification_problem(x,y): " << is_binary_classification_problem(x,y)
                << "\n\t basis_loaded(): " << basis_loaded()
                );

            if (ekm_stale)
            {
                ekm.load(kern, basis);
                ekm_stale = false;
            }

            // project all the samples with the ekm
            running_stats<scalar_type> rs;
            std::vector<matrix<scalar_type,0,1, mem_manager_type> > proj_samples;
            proj_samples.reserve(x.size());
            for (long i = 0; i < x.size(); ++i)
            {
                if (verbose)
                {
                    scalar_type err;
                    proj_samples.push_back(ekm.project(x(i), err));
                    rs.add(err);
                }
                else
                {
                    proj_samples.push_back(ekm.project(x(i)));
                }
            }

            if (verbose)
            {
                std::cout << "\nMean EKM projection error:                 " << rs.mean() << std::endl;
                std::cout << "Standard deviaion of EKM projection error: " << rs.stddev() << std::endl;
            }
            
            // now do the training
            decision_function<linear_kernel<matrix<scalar_type,0,1, mem_manager_type> > > df;
            df = ocas.train(proj_samples, y, svm_objective);

            if (verbose)
            {
                std::cout << "Final svm objective: " << svm_objective << std::endl;
            }

            decision_function<kernel_type> final_df;
            final_df = ekm.convert_to_decision_function(df.basis_vectors(0));
            final_df.b = df.b;
            return final_df;
        }

        /*!
            CONVENTION
                - if (ekm_stale) then
                    - kern or basis have changed since the last time
                      they were loaded into the ekm
        !*/

        svm_c_linear_trainer<linear_kernel<matrix<scalar_type,0,1,mem_manager_type> > > ocas;
        bool verbose;

        kernel_type kern;

        matrix<sample_type,0,1,mem_manager_type> basis;
        mutable empirical_kernel_map<kernel_type> ekm;
        mutable bool ekm_stale; 

    }; 

}

#endif // DLIB_SVM_C_EKm_TRAINER_H__



