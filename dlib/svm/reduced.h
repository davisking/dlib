// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_REDUCEd_TRAINERS_
#define DLIB_REDUCEd_TRAINERS_

#include "reduced_abstract.h"
#include "../matrix.h"
#include "../algs.h"
#include "function.h"
#include "kernel.h"
#include "kcentroid.h"
#include "linearly_independent_subset_finder.h"
#include "../optimization.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type 
        >
    class reduced_decision_function_trainer
    {
    public:
        typedef typename trainer_type::kernel_type kernel_type;
        typedef typename trainer_type::scalar_type scalar_type;
        typedef typename trainer_type::sample_type sample_type;
        typedef typename trainer_type::mem_manager_type mem_manager_type;
        typedef typename trainer_type::trained_function_type trained_function_type;

        explicit reduced_decision_function_trainer (
            const trainer_type& trainer_,
            const unsigned long num_sv_ 
        ) :
            trainer(trainer_),
            num_sv(num_sv_)
        {
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
            return do_train(vector_to_matrix(x), vector_to_matrix(y));
        }

    private:

    // ------------------------------------------------------------------------------------

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> do_train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y
        ) const
        {
            // get the decision function object we are going to try and approximate
            const decision_function<kernel_type> dec_funct = trainer.train(x,y);
            
            // now find a linearly independent subset of the training points of num_sv points.
            linearly_independent_subset_finder<kernel_type> lisf(trainer.get_kernel(), num_sv);
            for (long i = 0; i < x.nr(); ++i)
            {
                lisf.add(x(i));
            }

            // make num be the number of points in the lisf object.  Just do this so we don't have
            // to write out lisf.dictionary_size() all over the place.
            const long num = lisf.dictionary_size();

            // The next few blocks of code just find the best weights with which to approximate 
            // the dec_funct object with the smaller set of vectors in the lisf dictionary.  This
            // is really just a simple application of some linear algebra.  For the details 
            // see page 554 of Learning with kernels by Scholkopf and Smola where they talk 
            // about "Optimal Expansion Coefficients."
            matrix<scalar_type, 0, 0, mem_manager_type> K_inv(num, num); 
            matrix<scalar_type, 0, 0, mem_manager_type> K(num, dec_funct.alpha.size()); 

            const kernel_type kernel(trainer.get_kernel());

            for (long r = 0; r < K_inv.nr(); ++r)
            {
                for (long c = 0; c < K_inv.nc(); ++c)
                {
                    K_inv(r,c) = kernel(lisf[r], lisf[c]);
                }
            }
            K_inv = pinv(K_inv);


            for (long r = 0; r < K.nr(); ++r)
            {
                for (long c = 0; c < K.nc(); ++c)
                {
                    K(r,c) = kernel(lisf[r], dec_funct.support_vectors(c));
                }
            }


            // Now we compute the approximate decision function.  Note that the weights come out
            // of the expression K_inv*K*dec_funct.alpha.
            decision_function<kernel_type> new_df(K_inv*K*dec_funct.alpha, 
                                                  0,
                                                  kernel, 
                                                  lisf.get_dictionary());

            // now we have to figure out what the new bias should be.  It might be a little
            // different since we just messed with all the weights and vectors.
            double bias = 0;
            for (long i = 0; i < x.nr(); ++i)
            {
                bias += new_df(x(i)) - dec_funct(x(i));
            }
            
            new_df.b = bias/x.nr();

            return new_df;
        }

    // ------------------------------------------------------------------------------------

        const trainer_type& trainer;
        const unsigned long num_sv; 


    }; // end of class reduced_decision_function_trainer

    template <typename trainer_type>
    const reduced_decision_function_trainer<trainer_type> reduced (
        const trainer_type& trainer,
        const unsigned long num_sv
    )
    {
        return reduced_decision_function_trainer<trainer_type>(trainer, num_sv);
    }

// ----------------------------------------------------------------------------------------


    template <
        typename trainer_type 
        >
    class reduced_decision_function_trainer2
    {
    public:
        typedef typename trainer_type::kernel_type kernel_type;
        typedef typename trainer_type::scalar_type scalar_type;
        typedef typename trainer_type::sample_type sample_type;
        typedef typename trainer_type::mem_manager_type mem_manager_type;
        typedef typename trainer_type::trained_function_type trained_function_type;

        explicit reduced_decision_function_trainer2 (
            const trainer_type& trainer_,
            const long num_ 
        ) :
            trainer(trainer_),
            num(num_)
        {
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
            return do_train(vector_to_matrix(x), vector_to_matrix(y));
        }

    private:

    // ------------------------------------------------------------------------------------

        class objective
        {
        public:
            objective(
                const decision_function<kernel_type>& dec_funct_,
                const matrix<scalar_type,0,1,mem_manager_type>& b_,
                const std::vector<sample_type>& out_vectors_
            ) :
                dec_funct(dec_funct_),
                b(b_),
                out_vectors(out_vectors_)
            {}

            double operator() (
                const sample_type& z
            ) const
            {
                // compute and return the error between the vector represented
                // by the decision_function and the vector represented by
                // the combination of the out_vectors and b objects.
                const double kzz = dec_funct.kernel_function(z,z); 

                double temp = dec_funct(z)+dec_funct.b;
                for (long i = 0; i < b.size(); ++i)
                    temp -= b(i)*dec_funct.kernel_function(z,out_vectors[i]);

                return -temp*temp/kzz;
            }

        private:

            const decision_function<kernel_type>& dec_funct;
            const matrix<scalar_type,0,1,mem_manager_type>& b;
            const std::vector<sample_type>& out_vectors;

        };

    // ------------------------------------------------------------------------------------

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> do_train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y
        ) const
        {
            using namespace std;

            matrix<scalar_type,0,0,mem_manager_type> K_inv, K;
            matrix<scalar_type,0,1,mem_manager_type> a, b, k;
            std::vector<sample_type> out_vectors;

            decision_function<kernel_type> dec_funct = trainer.train(x,y);
            dlib::rand::kernel_1a rnd;
            sample_type sample, best_sample;

            const long num_in_vects = dec_funct.support_vectors.nr();


            const objective obj(dec_funct, b, out_vectors);
            const kernel_type kernel(dec_funct.kernel_function);

            // find the minimum possilbe value of the objective
            double min_obj = -1; // add a -1 here instead of 0 for good measure
            for (long i = 0; i < num_in_vects; ++i)
            {
                for (long j = 0; j < num_in_vects; ++j)
                {
                    min_obj -= dec_funct.alpha(i)*dec_funct.alpha(j)*
                                kernel(dec_funct.support_vectors(i), dec_funct.support_vectors(j));
                }
            }

            cout << "min_obj: " << min_obj << endl;

            for (long i = 0; i < num; ++i)
            {
                double best_error = std::numeric_limits<double>::max();
                // select the next vector 
                for (long j = 0; j < 6; ++j)
                {
                    // pick a random vector from the decision function
                    const long random_index = rnd.get_random_32bit_number()%num_in_vects;
                    sample = dec_funct.support_vectors(random_index);
                    find_min_conjugate_gradient2(obj, sample, min_obj ); 

                    const double obj_value = obj(sample);
                    cout << "obj(sample): " << obj_value << endl;
                    if (obj_value < best_error)
                    {
                        best_error = obj_value;
                        best_sample = sample;
                    }
                }

                cout << "best_sample:  " << trans(best_sample);
                const scalar_type k_bs = kernel(best_sample,best_sample);

                // Now we need to update the K and K_inv matrices 
                if (i == 0)
                {
                    K_inv.set_size(1,1);
                    K_inv(0,0) = 1/k_bs;

                    K.set_size(1,num_in_vects);
                    for (long c = 0; c < K.nc(); ++c)
                        K(K.nr()-1,c) = kernel(best_sample, dec_funct.support_vectors(c));
                }
                else
                {
                    // fill in k
                    k.set_size(out_vectors.size());
                    for (long r = 0; r < k.nr(); ++r)
                        k(r) = kernel(best_sample,out_vectors[r]);

                    // compute the error we would have if we approximated the new x sample
                    // with the dictionary.  That is, do the ALD test from the KRLS paper.
                    a = K_inv*k;
                    const scalar_type delta = k_bs - trans(k)*a;

                    using namespace std;
                    cout << "delta: " << delta << endl;
                    if (delta > std::numeric_limits<scalar_type>::epsilon())
                    {

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


                        // now update the K matrix
                        temp.set_size(K.nr()+1, K.nc());
                        // copy over the old K matrix into the top of temp
                        set_subm(temp, get_rect(K)) = K;
                        // put temp into K
                        temp.swap(K);
                        for (long c = 0; c < K.nc(); ++c)
                            K(K.nr()-1,c) = kernel(best_sample, dec_funct.support_vectors(c));
                    }
                    else
                    {
                        // this last vector we tried to add is a linear combination of the 
                        // previous vectors so we should just stop now since there isn't
                        // any point in adding any more vectors.
                        break;
                    }

                }

                // now that we have updated the K and K_inv matrices we can update
                // the b vector.
                b = K_inv*K*dec_funct.alpha;

                // also record this new vector
                out_vectors.push_back(best_sample);
            }

            decision_function<kernel_type> new_df(b, 
                                                  0,
                                                  kernel, 
                                                  vector_to_matrix(out_vectors));

            // now we have to figure out what the new bias should be
            double bias = 0;
            for (long i = 0; i < x.nr(); ++i)
            {
                bias += new_df(x(i)) - dec_funct(x(i));
            }
            
            new_df.b = bias/x.nr();

            return new_df;
        }

        const trainer_type& trainer;
        const long num;


    }; // end of class reduced_decision_function_trainer2

    template <typename trainer_type>
    const reduced_decision_function_trainer2<trainer_type> reduced2 (
        const trainer_type& trainer,
        const long num 
    )
    {
        return reduced_decision_function_trainer2<trainer_type>(trainer, num);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_REDUCEd_TRAINERS_

