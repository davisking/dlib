// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PEGASoS_
#define DLIB_PEGASoS_

#include "pegasos_abstract.h"
#include <cmath>
#include "../algs.h"
#include "function.h"
#include "kernel.h"
#include "kcentroid.h"
#include <iostream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename K 
        >
    class svm_pegasos
    {
        typedef kcentroid<offset_kernel<K> > kc_type;

    public:
        typedef K kernel_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef decision_function<kernel_type> trained_function_type;

        svm_pegasos (
        ) :
            lambda(0.0001),
            tau(0.01),
            tolerance(0.01),
            train_count(0),
            w(offset_kernel<kernel_type>(kernel,tau),tolerance)
        {
        }

        svm_pegasos (
            const kernel_type& kernel_, 
            const scalar_type& lambda_,
            const scalar_type& tolerance_
        ) :
            kernel(kernel_),
            lambda(lambda_),
            tau(0.01),
            tolerance(tolerance_),
            train_count(0),
            w(offset_kernel<kernel_type>(kernel,tau),tolerance)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(lambda > 0 && tolerance > 0,
                        "\tsvm_pegasos::svm_pegasos(kernel,lambda,tolerance)"
                        << "\n\t invalid inputs were given to this function"
                        << "\n\t lambda: " << lambda 
            );
        }

        void clear (
        )
        {
            // reset the w vector back to its initial state
            w = kc_type(offset_kernel<kernel_type>(kernel,tau),tolerance);
            train_count = 0;
        }

        void set_kernel (
            kernel_type k
        )
        {
            kernel = k;
            clear();
        }

        void set_tolerance (
            double tol
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(0 < tol,
                        "\tvoid svm_pegasos::set_tolerance(tol)"
                        << "\n\t invalid inputs were given to this function"
                        << "\n\t tol: " << tol 
            );
            tolerance = tol;
            clear();
        }

        void set_lambda (
            scalar_type lambda_
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(0 < lambda_,
                        "\tvoid svm_pegasos::set_lambda(lambda_)"
                        << "\n\t invalid inputs were given to this function"
                        << "\n\t lambda: " << lambda_ 
            );
            lambda = lambda_;
            clear();
        }

        const scalar_type get_lambda (
        ) const
        {
            return lambda;
        }

        const scalar_type get_tolerance (
        ) const
        {
            return tolerance;
        }

        const kernel_type get_kernel (
        ) const
        {
            return kernel;
        }

        unsigned long get_train_count (
        ) const
        {
            return static_cast<unsigned long>(train_count);
        }

        scalar_type train (
            const sample_type& x,
            const scalar_type& y
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(y == -1 || y == 1,
                        "\tscalar_type svm_pegasos::train(x,y)"
                        << "\n\t invalid inputs were given to this function"
                        << "\n\t y: " << y
            );
            ++train_count;
            const scalar_type learning_rate = 1/(lambda*train_count);

            // if this sample point is within the margin of the current hyperplane
            if (y*w.inner_product(x) < 1)
            {

                // compute: w = (1-learning_rate*lambda)*w + y*learning_rate*x
                w.train(x,  1 - learning_rate*lambda,  y*learning_rate);

                scalar_type wnorm = std::sqrt(w.squared_norm());
                scalar_type temp = (1/std::sqrt(lambda))/(wnorm);
                if (temp < 1)
                    w.scale_by(temp);
            }
            else
            {
                w.scale_by(1 - learning_rate*lambda);
            }

            return learning_rate;
        }

        scalar_type operator() (
            const sample_type& x
        ) const
        {
            return w.inner_product(x);
        }

        const decision_function<kernel_type> get_decision_function (
        ) const
        {
            distance_function<offset_kernel<kernel_type> > df = w.get_distance_function();
            return decision_function<kernel_type>(df.alpha, -tau*sum(df.alpha), kernel, df.support_vectors);
        }

        void swap (
            svm_pegasos& item
        )
        {
            exchange(kernel,         item.kernel);
            exchange(lambda,         item.lambda);
            exchange(tau,            item.tau);
            exchange(tolerance,      item.tolerance);
            exchange(train_count,    item.train_count);
            exchange(w,              item.w);
        }

        friend void serialize(const svm_pegasos& item, std::ostream& out)
        {
            serialize(item.kernel, out);
            serialize(item.lambda, out);
            serialize(item.tau, out);
            serialize(item.tolerance, out);
            serialize(item.train_count, out);
            serialize(item.w, out);
        }

        friend void deserialize(svm_pegasos& item, std::istream& in)
        {
            deserialize(item.kernel, in);
            deserialize(item.lambda, in);
            deserialize(item.tau, in);
            deserialize(item.tolerance, in);
            deserialize(item.train_count, in);
            deserialize(item.w, in);
        }

    private:

        kernel_type kernel;
        scalar_type lambda;
        scalar_type tau;
        scalar_type tolerance;
        scalar_type train_count;
        kc_type w;

    }; // end of class svm_pegasos

    template <
        typename K 
        >
    void swap (
        svm_pegasos<K>& a,
        svm_pegasos<K>& b
    ) { a.swap(b); }

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
        ) :
            min_learning_rate(0.1)
        {
        }

        batch_trainer (
            const trainer_type& trainer_, 
            const scalar_type min_learning_rate_,
            bool verbose_
        ) :
            trainer(trainer_),
            min_learning_rate(min_learning_rate_),
            verbose(verbose_)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(0 < min_learning_rate_,
                        "\tbatch_trainer::batch_trainer()"
                        << "\n\t invalid inputs were given to this function"
                        << "\n\t min_learning_rate_: " << min_learning_rate_ 
            );
        }

        const kernel_type get_kernel (
        ) const
        {
            return trainer.get_kernel();
        }

        const scalar_type get_min_learning_rate (
        ) const 
        {
            return min_learning_rate;
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

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> do_train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y
        ) const
        {
            typedef typename decision_function<kernel_type>::sample_vector_type sample_vector_type;
            typedef typename decision_function<kernel_type>::scalar_vector_type scalar_vector_type;

            dlib::rand::kernel_1a rnd;

            trainer_type my_trainer(trainer);

            scalar_type cur_learning_rate = min_learning_rate + 10;
            unsigned long count = 0;

            while (cur_learning_rate > min_learning_rate)
            {
                const long i = rnd.get_random_32bit_number()%x.size();
                // keep feeding the trainer data until its learning rate goes below our threshold
                cur_learning_rate = my_trainer.train(x(i), y(i));

                if (verbose)
                {
                    if ( (count&0x7FF) == 0)
                    {
                        std::cout << "\rrbatch_trainer(): Percent complete: " 
                                  << 100*min_learning_rate/cur_learning_rate << "             " << std::flush;
                    }
                    ++count;
                }
            }

            if (verbose)
            {
                decision_function<kernel_type> df = my_trainer.get_decision_function();
                std::cout << "\rbatch_trainer(): Percent complete: 100           " << std::endl;
                std::cout << "    Num sv: " << df.support_vectors.size() << std::endl;
                std::cout << "    bias:   " << df.b << std::endl;
                return df;
            }
            else
            {
                return my_trainer.get_decision_function();
            }
        }


        trainer_type trainer;
        scalar_type min_learning_rate;
        bool verbose;

    }; // end of class batch_trainer

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type
        >
    const batch_trainer<trainer_type> batch (
        const trainer_type& trainer,
        const typename trainer_type::scalar_type min_learning_rate = 0.1
    ) { return batch_trainer<trainer_type>(trainer, min_learning_rate, false); }

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type
        >
    const batch_trainer<trainer_type> verbose_batch (
        const trainer_type& trainer,
        const typename trainer_type::scalar_type min_learning_rate = 0.1
    ) { return batch_trainer<trainer_type>(trainer, min_learning_rate, true); }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PEGASoS_

