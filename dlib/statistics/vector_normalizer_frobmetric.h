// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_VECTOR_NORMALIZER_FRoBMETRIC_H__
#define DLIB_VECTOR_NORMALIZER_FRoBMETRIC_H__

#include "vector_normalizer_frobmetric_abstract.h"
#include "../matrix.h"
#include "../optimization.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type
        >
    struct frobmetric_training_sample 
    {
        matrix_type anchor;
        std::vector<matrix_type> near;
        std::vector<matrix_type> far;

        unsigned long num_triples (
        ) const { return near.size() * far.size(); }

        void clear()
        {
            near.clear();
            far.clear();
        }
    };

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type
        >
    class vector_normalizer_frobmetric
    {

    public:
        typedef typename matrix_type::mem_manager_type mem_manager_type;
        typedef typename matrix_type::type scalar_type;
        typedef matrix_type result_type;

    private:
        struct compact_frobmetric_training_sample 
        {
            std::vector<matrix_type> near;
            std::vector<matrix_type> far;
        };

        struct objective
        {
            objective (
                const std::vector<compact_frobmetric_training_sample>& samples_,
                matrix<double,0,0,mem_manager_type>& Aminus_
            ) : samples(samples_), Aminus(Aminus_) {}
            
            double operator()(const matrix<double,0,1,mem_manager_type>& u) const
            {
                long idx = 0;
                const long dims = samples[0].far[0].size();
                // Here we compute \hat A from the paper, which we refer to as just A in
                // the code.  
                matrix<double,0,0,mem_manager_type> A(dims,dims);
                A = 0;
                std::vector<double> ufar, unear;
                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    ufar.assign(samples[i].far.size(),0);
                    unear.assign(samples[i].near.size(),0);
                    for (unsigned long j = 0; j < unear.size(); ++j)
                    {
                        for (unsigned long k = 0; k < ufar.size(); ++k)
                        {
                            const double val = u(idx++);
                            ufar[k] -= val;
                            unear[j] += val;
                        }
                    }
                    for (unsigned long j = 0; j < unear.size(); ++j)
                        A += unear[j]*samples[i].near[j]*trans(samples[i].near[j]);
                    for (unsigned long j = 0; j < ufar.size(); ++j)
                        A += ufar[j]*samples[i].far[j]*trans(samples[i].far[j]);
                }

                eigenvalue_decomposition<matrix<double,0,0,mem_manager_type> > ed(make_symmetric(A));
                Aminus = ed.get_pseudo_v()*diagm(upperbound(ed.get_real_eigenvalues(),0))*trans(ed.get_pseudo_v());
                // Do this to avoid numeric instability later on since the above
                // computation can make Aminus slightly non-symmetric.
                Aminus = make_symmetric(Aminus);

                return sum(u) - 0.5*sum(squared(Aminus));
            }

        private:
            const std::vector<compact_frobmetric_training_sample>& samples;
            matrix<double,0,0,mem_manager_type>& Aminus;
        };

        struct derivative
        {
            derivative (
                unsigned long num_triples_,
                const std::vector<compact_frobmetric_training_sample>& samples_,
                matrix<double,0,0,mem_manager_type>& Aminus_
            ) : num_triples(num_triples_), samples(samples_), Aminus(Aminus_) {}
            
            matrix<double,0,1,mem_manager_type> operator()(const matrix<double,0,1,mem_manager_type>& ) const
            {
                // Note that Aminus is a function of u (the argument to this function), but
                // since Aminus will have been computed already by the most recent call to
                // the objective function we don't need to do anything with u.  We can just
                // use Aminus right away.
                matrix<double,0,1,mem_manager_type> grad(num_triples);
                
                long idx = 0;
                std::vector<double> ufar, unear;
                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    ufar.resize(samples[i].far.size());
                    unear.resize(samples[i].near.size());

                    for (unsigned long j = 0; j < unear.size(); ++j)
                        unear[j] = sum(pointwise_multiply(Aminus, samples[i].near[j]*trans(samples[i].near[j])));
                    for (unsigned long j = 0; j < ufar.size(); ++j)
                        ufar[j] = sum(pointwise_multiply(Aminus, samples[i].far[j]*trans(samples[i].far[j])));

                    for (unsigned long j = 0; j < samples[i].near.size(); ++j)
                    {
                        for (unsigned long k = 0; k < samples[i].far.size(); ++k)
                        {
                            grad(idx++) = 1 + ufar[k]-unear[j];
                        }
                    }
                }

                return grad;
            }

        private:
            const unsigned long num_triples;
            const std::vector<compact_frobmetric_training_sample>& samples;
            matrix<double,0,0,mem_manager_type>& Aminus;
        };


        class custom_stop_strategy
        {
        public:
            custom_stop_strategy(
                double C_,
                double eps_,
                bool be_verbose_
            ) 
            {
                _C = C_;

                _cur_iter = 0;
                _gradient_thresh = eps_;
                _max_iter = 1000;
                _verbose = be_verbose_;
            }

            template <typename T>
            bool should_continue_search (
                const T& u,
                const double ,
                const T& grad
            ) 
            {
                ++_cur_iter;

                double max_gradient = 0;
                for (long i = 0; i < grad.size(); ++i)
                {
                    const bool at_lower_bound = (0 >= u(i) && grad(i) > 0);
                    const bool at_upper_bound = (_C/grad.size() <= u(i) && grad(i) < 0);
                    if (!at_lower_bound && !at_upper_bound)
                        max_gradient = std::max(std::abs(grad(i)), max_gradient);
                }

                if (_verbose)
                {
                    std::cout << "iteration: " << _cur_iter << "   max_gradient: "<< max_gradient << std::endl;
                }

                // Only stop when the largest non-bound-constrained element of the gradient
                // is lower than the threshold.
                if (max_gradient < _gradient_thresh)
                    return false;

                // Check if we have hit the max allowable number of iterations.  
                if (_cur_iter > _max_iter)
                {
                    return false;
                }

                return true;
            }

        private:
            bool _verbose;

            unsigned long _max_iter;
            unsigned long _cur_iter;
            double _C;
            double _gradient_thresh;
        };

    public:
        vector_normalizer_frobmetric (
        )
        {
            verbose = false;
            eps = 0.1;
            C = 1;
        }

        void be_verbose(
        )
        {
            verbose = true;
        }

        void set_epsilon (
            double eps_
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(eps_ > 0,
                "\t void vector_normalizer_frobmetric::set_epsilon(eps_)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t eps: " << eps_ 
                );
            eps = eps_;
        }

        double get_epsilon (
        ) const 
        {
            return eps;
        }

        void set_c (
            double C_ 
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(C_ > 0,
                "\t void vector_normalizer_frobmetric::set_c()"
                << "\n\t C_ must be greater than 0"
                << "\n\t C_:    " << C_ 
                << "\n\t this: " << this
                );

            C = C_;
        }

        double get_c (
        ) const
        {
            return C;
        }

        void be_quiet (
        )
        {
            verbose = false;
        }
       
        void train (
            const std::vector<frobmetric_training_sample<matrix_type> >& samples
        )
        /*!
            requires
                - samples.size() != 0
                - All matrices inside samples (i.e. anchors and elements of near and far)
                  are column vectors with the same non-zero dimension.
                - All the vectors in samples contain finite values.
                - All elements of samples contain data, specifically, for all valid i:
                    - samples[i].num_triples() != 0
        !*/
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(samples.size() > 0,
                "\tvoid vector_normalizer_frobmetric::train()"
                << "\n\t you have to give a nonempty set of samples to this function"
                );
#ifdef ENABLE_ASSERTS
            {
                const long dims = samples[0].anchor.size();
                DLIB_ASSERT(dims != 0,
                    "\tvoid vector_normalizer_frobmetric::train()"
                    << "\n\t The dimension of the input vectors can't be zero."
                    );
                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    DLIB_ASSERT(is_col_vector(samples[i].anchor), 
                        "\tvoid vector_normalizer_frobmetric::train()"
                        << "\n\t Invalid inputs were given to this function."
                        << "\n\t i: " << i
                        );
                    DLIB_ASSERT(samples[i].anchor.size() == dims, 
                        "\tvoid vector_normalizer_frobmetric::train()"
                        << "\n\t Invalid inputs were given to this function."
                        << "\n\t i:    " << i
                        << "\n\t dims: " << dims
                        << "\n\t samples[i].anchor.size(): " << samples[i].anchor.size()
                        );

                    DLIB_ASSERT(samples[i].num_triples() != 0,
                        "\tvoid vector_normalizer_frobmetric::train()"
                        << "\n\t It is illegal for a training sample to have no data in it"
                        << "\n\t i:    " << i
                    );
                    for (unsigned long j = 0; j < samples[i].near.size(); ++j)
                    {
                        DLIB_ASSERT(is_col_vector(samples[i].near[j]), 
                            "\tvoid vector_normalizer_frobmetric::train()"
                            << "\n\t Invalid inputs were given to this function."
                            << "\n\t i: " << i
                            << "\n\t j: " << j
                            );
                        DLIB_ASSERT(samples[i].near[j].size() == dims, 
                            "\tvoid vector_normalizer_frobmetric::train()"
                            << "\n\t Invalid inputs were given to this function."
                            << "\n\t i:    " << i
                            << "\n\t j:    " << j
                            << "\n\t dims: " << dims
                            << "\n\t samples[i].near[j].size(): " << samples[i].near[j].size()
                            );
                    }
                    for (unsigned long j = 0; j < samples[i].far.size(); ++j)
                    {
                        DLIB_ASSERT(is_col_vector(samples[i].far[j]), 
                            "\tvoid vector_normalizer_frobmetric::train()"
                            << "\n\t Invalid inputs were given to this function."
                            << "\n\t i: " << i
                            << "\n\t j: " << j
                            );
                        DLIB_ASSERT(samples[i].far[j].size() == dims, 
                            "\tvoid vector_normalizer_frobmetric::train()"
                            << "\n\t Invalid inputs were given to this function."
                            << "\n\t i:    " << i
                            << "\n\t j:    " << j
                            << "\n\t dims: " << dims
                            << "\n\t samples[i].far[j].size(): " << samples[i].far[j].size()
                            );
                    }
                }
            }
#endif // end ENABLE_ASSERTS


            // compute the mean sample
            m = 0;
            for (unsigned long i = 0; i < samples.size(); ++i)
                m += samples[i].anchor;
            m /= samples.size();

            DLIB_ASSERT(is_finite(m), "Some of the input vectors to vector_normalizer_frobmetric::train() have infinite or NaN values");

            // Now we need to find tform.  So we setup the optimization problem and run it
            // over the next few lines of code.
            unsigned long num_triples = 0;
            for (unsigned long i = 0; i < samples.size(); ++i)
                num_triples += samples[i].near.size()*samples[i].far.size();

            matrix<double,0,1,mem_manager_type> u(num_triples);
            u = 0;


            // precompute all the anchor to far/near pairs
            std::vector<compact_frobmetric_training_sample> data(samples.size());
            for (unsigned long i = 0; i < data.size(); ++i)
            {
                data[i].far.reserve(samples[i].far.size());
                data[i].near.reserve(samples[i].near.size());
                for (unsigned long j = 0; j < samples[i].far.size(); ++j)
                    data[i].far.push_back(samples[i].anchor - samples[i].far[j]);
                for (unsigned long j = 0; j < samples[i].near.size(); ++j)
                    data[i].near.push_back(samples[i].anchor - samples[i].near[j]);
            }

            // Now run the main part of the algorithm
            matrix<double,0,0,mem_manager_type> Aminus;
            find_max_box_constrained(lbfgs_search_strategy(10),
                                     custom_stop_strategy(C, eps, verbose),
                                     objective(data, Aminus),
                                     derivative(num_triples, data, Aminus),
                                     u, 0, C/num_triples);


            // What we need is the optimal Aminus which is a function of u.  So we already
            // have what we need and just need to put it into tform.
            eigenvalue_decomposition<matrix<double,0,0,mem_manager_type> > ed(make_symmetric(-Aminus));
            matrix<double,0,1,mem_manager_type> eigs = ed.get_real_eigenvalues();
            // But first, discard the components that are zero to within the machine epsilon.
            const double tol = max(eigs)*std::numeric_limits<double>::epsilon();
            for (long i = 0; i < eigs.size(); ++i)
            {
                if (eigs(i) < tol)
                    eigs(i) = 0;
            }
            tform = matrix_cast<scalar_type>(diagm(sqrt(eigs))*trans(ed.get_pseudo_v()));

            // Pre-apply the transform to m so we don't have to do it inside operator()
            // every time it's called.
            m = tform*m;
        }

        long in_vector_size (
        ) const
        {
            return m.nr();
        }

        long out_vector_size (
        ) const
        {
            return m.nr();
        }

        const matrix<scalar_type,0,1,mem_manager_type>& transformed_means (
        ) const
        {
            return m;
        }

        const matrix<scalar_type,0,0,mem_manager_type>& transform (
        ) const
        {
            return tform;
        }

        const result_type& operator() (
            const matrix_type& x
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(in_vector_size() != 0 && in_vector_size() == x.size() && 
                is_col_vector(x) == true,
                "\tmatrix vector_normalizer_frobmetric::operator()"
                << "\n\t you have given invalid arguments to this function"
                << "\n\t in_vector_size(): " << in_vector_size()
                << "\n\t x.size():         " << x.size()
                << "\n\t is_col_vector(x): " << is_col_vector(x)
                << "\n\t this:             " << this
                );

            temp_out = tform*x-m;
            return temp_out;
        }

        template <typename mt>
        friend void deserialize (
            vector_normalizer_frobmetric<mt>& item, 
            std::istream& in
        ); 

        template <typename mt>
        friend void serialize (
            const vector_normalizer_frobmetric<mt>& item, 
            std::ostream& out 
        );

    private:

        // ------------------- private data members -------------------

        matrix_type m;
        matrix<scalar_type,0,0,mem_manager_type> tform;
        bool verbose;
        double eps;
        double C;

        // This is just a temporary variable that doesn't contribute to the
        // state of this object.
        mutable matrix_type temp_out;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type
        >
    void serialize (
        const vector_normalizer_frobmetric<matrix_type>& item, 
        std::ostream& out 
    )
    {
        const int version = 1;
        serialize(version, out);

        serialize(item.m, out);
        serialize(item.tform, out);
        serialize(item.verbose, out);
        serialize(item.eps, out);
        serialize(item.C, out);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type
        >
    void deserialize (
        vector_normalizer_frobmetric<matrix_type>& item, 
        std::istream& in
    )   
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unsupported version found while deserializing dlib::vector_normalizer_frobmetric.");

        deserialize(item.m, in);
        deserialize(item.tform, in);
        deserialize(item.verbose, in);
        deserialize(item.eps, in);
        deserialize(item.C, in);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_VECTOR_NORMALIZER_FRoBMETRIC_H__

