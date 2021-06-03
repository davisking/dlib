// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_ElASTIC_NET_Hh_
#define DLIB_ElASTIC_NET_Hh_

#include "../matrix.h"
#include "elastic_net_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class elastic_net
    {
    public:

        template <typename EXP>
        explicit elastic_net(
            const matrix_exp<EXP>& XX
        ) : eps(1e-5), max_iterations(50000), verbose(false)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(XX.size() > 0 &&
                        XX.nr() == XX.nc(),
                "\t elastic_net::elastic_net(XX)"
                << " \n\t XX must be a non-empty square matrix."
                << " \n\t XX.nr():   " << XX.nr() 
                << " \n\t XX.nc():   " << XX.nc() 
                << " \n\t this: " << this
                );


            // If the number of columns in X is big and in particular bigger than the number of
            // rows then we can get rid of them by doing some SVD magic.  Doing this doesn't
            // make the final results of anything change but makes all the matrices have
            // dimensions that are X.nr() in size, which can be much smaller.
            matrix<double,0,1> s;
            svd3(XX,u,eig_vals,eig_vects);
            s = sqrt(eig_vals);
            X = eig_vects*diagm(s);
            u = eig_vects*inv(diagm(s));



            samples.resize(X.nr()*2);

            for (size_t i = 0; i < samples.size(); ++i)
                index.push_back(i);
            active_size = index.size();


            // setup the training samples used in the SVM optimizer below
            for (size_t i = 0; i < samples.size(); ++i)
            {
                auto& x = samples[i];
                const long idx = i/2;
                if (i%2 == 0)
                    x.label = +1;
                else
                    x.label = -1;

                x.r = idx%X.nr();
            }
        }

        template <typename EXP1, typename EXP2>
        elastic_net(
            const matrix_exp<EXP1>& XX,
            const matrix_exp<EXP2>& XY
        ) : elastic_net(XX)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(XX.size() > 0 && 
                        XX.nr() == XX.nc() &&
                        is_col_vector(XY) && 
                        XX.nc() == XY.size() ,
                "\t elastic_net::elastic_net(XX,XY)"
                << " \n\t Invalid inputs were given to this function."
                << " \n\t XX.size(): " << XX.size() 
                << " \n\t is_col_vector(XY): " << is_col_vector(XY) 
                << " \n\t XX.nr():   " << XX.nr() 
                << " \n\t XX.nc():   " << XX.nc() 
                << " \n\t XY.size(): " << XY.size() 
                << " \n\t this: " << this
                );

            set_xy(XY);
        }

        long size (
        ) const { return u.nr(); }

        template <typename EXP>
        void set_xy(
            const matrix_exp<EXP>& XY
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_col_vector(XY) && 
                        XY.size() == size(),
                "\t void elastic_net::set_y(Y)"
                << " \n\t Invalid inputs were given to this function."
                << " \n\t is_col_vector(XY): " << is_col_vector(XY) 
                << " \n\t size():    " << size() 
                << " \n\t XY.size(): " << XY.size() 
                << " \n\t this: " << this
                );

            Y = trans(u)*XY;
            // We can use the ynorm after it has been projected because the only place Y
            // appears in the algorithm is in terms of dot products with w and x vectors.
            // But those vectors are always in the span of X and therefore we only see the
            // part of the norm of Y that is in the span of X (and hence u since u and X
            // have the same span by construction)
            ynorm = length_squared(Y); 
            xdoty = X*Y;
            eig_vects_xdoty = trans(eig_vects)*xdoty;

            w.set_size(Y.size());
            // zero out any memory of previous solutions
            alpha.assign(X.nr()*2, 0);
        }

        bool have_target_values (
        ) const { return Y.size() != 0; }

        void set_epsilon(
            double eps_
        ) 
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(eps_ > 0,
                "\t void elastic_net::set_epsilon()"
                << " \n\t eps_ must be greater than 0"
                << " \n\t eps_: " << eps_ 
                << " \n\t this: " << this
                );

            eps = eps_;
        }

        unsigned long get_max_iterations (
        ) const { return max_iterations; }

        void set_max_iterations (
            unsigned long max_iter
        ) 
        {
            max_iterations = max_iter;
        }

        void be_verbose (
        )
        {
            verbose = true;
        }

        void be_quiet (
        )
        {
            verbose = false;
        }

        double get_epsilon (
        ) const { return eps; }

        matrix<double,0,1> operator() (
            double ridge_lambda,
            double lasso_budget = std::numeric_limits<double>::infinity()
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(have_target_values() && 
                        ridge_lambda > 0 && 
                        lasso_budget > 0 ,
                "\t matrix<double,0,1> elastic_net::operator()()"
                << " \n\t Invalid inputs were given to this function."
                << " \n\t have_target_values(): " << have_target_values() 
                << " \n\t ridge_lambda: " << ridge_lambda 
                << " \n\t lasso_budget: " << lasso_budget 
                << " \n\t this: " << this
                );


            // First check if lasso_budget is so big that it isn't even active.  We do this
            // by doing just ridge regression and checking the result.
            matrix<double,0,1> betas = eig_vects*tmp(inv(diagm(eig_vals + ridge_lambda))*eig_vects_xdoty);
            if (sum(abs(betas)) <= lasso_budget)
                return betas;


            // Set w back to 0.  We will compute the w corresponding to what is currently
            // in alpha layer on.  This way w and alpha are always in sync.
            w = 0;
            wy_mult = 0;
            wdoty = 0;


            // return dot(w,x)
            auto dot = [&](const matrix<double,0,1>& w, const en_sample2& x)
            {
                const double xmul = -x.label*(1/lasso_budget);
                // Do the base dot product but don't forget to add in the -(1/t)*y part from the svm reduction paper
                double val = rowm(X,x.r)*w + xmul*wdoty + wy_mult*xdoty(x.r) + xmul*wy_mult*ynorm;

                return val;
            };


            // perform w += scale*x;
            auto add_to = [&](matrix<double,0,1>& w, double scale, const en_sample2& x)
            {
                const double xmul = -x.label*(1/lasso_budget);
                wy_mult += scale*xmul;
                wdoty += scale*xdoty(x.r);
                w += scale*trans(rowm(X,x.r));

            };

            const double Dii = ridge_lambda;

            // setup the training samples used in the SVM optimizer below
            for (size_t i = 0; i < samples.size(); ++i)
            {
                auto& x = samples[i];

                const double xmul = -x.label*(1/lasso_budget);
                x.xdotx = xmul*xmul*ynorm;
                for (long c = 0; c < X.nc(); ++c)
                    x.xdotx += std::pow(X(x.r,c)+xmul*Y(c), 2.0) - std::pow(xmul*Y(c),2.0);

                // compute the correct w given whatever might be in alpha.
                if (alpha[i] != 0)
                    add_to(w, x.label*alpha[i], samples[i]);
            }


            // Now run the optimizer
            double PG_max_prev = std::numeric_limits<double>::infinity();
            double PG_min_prev = -std::numeric_limits<double>::infinity();


            unsigned int iter;
            for (iter = 0; iter < max_iterations; ++iter)
            {
                // randomly shuffle the indices
                for (unsigned long i = 0; i < active_size; ++i)
                {
                    // pick a random index >= i
                    const long j = i + rnd.get_random_32bit_number()%(active_size-i);
                    std::swap(index[i], index[j]);
                }

                double PG_max = -std::numeric_limits<double>::infinity();
                double PG_min = std::numeric_limits<double>::infinity();
                for (size_t ii = 0; ii < active_size; ++ii)
                {
                    const auto i = index[ii];
                    const auto& x = samples[i];
                    double G = x.label*dot(w, x) - 1 + Dii*alpha[i];

                    double PG = 0;
                    if (alpha[i] == 0)
                    {
                        if (G > PG_max_prev)
                        {
                            // shrink the active set of training examples
                            --active_size;
                            std::swap(index[ii], index[active_size]);
                            --ii;
                            continue;
                        }

                        if (G < 0)
                            PG = G;
                    }
                    else
                    {
                        PG = G;
                    }

                    if (PG > PG_max) 
                        PG_max = PG;
                    if (PG < PG_min) 
                        PG_min = PG;

                    // if PG != 0
                    if (std::abs(PG) > 1e-12)
                    {
                        const double alpha_old = alpha[i];
                        alpha[i] = std::max(alpha[i] - G/(x.xdotx+Dii), (double)0.0);
                        const double delta = (alpha[i]-alpha_old)*x.label;
                        add_to(w, delta, x);
                    }
                }

                if (verbose)
                {
                    using namespace std;
                    cout << "gap:         " << PG_max - PG_min << endl;
                    cout << "active_size: " << active_size << endl;
                    cout << "iter:        " << iter << endl;
                    cout << endl;
                }

                if (PG_max - PG_min <= eps)
                {
                    // stop if we are within eps tolerance and the last iteration
                    // was over all the samples
                    if (active_size == index.size())
                        break;

                    // Turn off shrinking on the next iteration.  We will stop if the
                    // tolerance is still <= eps when shrinking is off.
                    active_size = index.size();
                    PG_max_prev = std::numeric_limits<double>::infinity();
                    PG_min_prev = -std::numeric_limits<double>::infinity();
                }
                else
                {
                    PG_max_prev = PG_max;
                    PG_min_prev = PG_min;
                    if (PG_max_prev <= 0)
                        PG_max_prev = std::numeric_limits<double>::infinity();
                    if (PG_min_prev >= 0)
                        PG_min_prev = -std::numeric_limits<double>::infinity();
                }


                // recalculate wdoty every so often to avoid drift.
                if (iter%100 == 0)
                    wdoty = dlib::dot(Y, w);
            }


            betas.set_size(alpha.size()/2);
            for (long i = 0; i < betas.size(); ++i)
                betas(i) = lasso_budget*(alpha[2*i] - alpha[2*i+1]);
            betas /= sum(mat(alpha));
            return betas;
        }


    private:

        struct en_sample2
        {
            // X location
            long r;


            double label;

            double xdotx;
        };

        std::vector<en_sample2> samples;
        std::vector<double> alpha;
        double ynorm;
        matrix<double> X;
        matrix<double,0,1> Y;
        matrix<double,0,1> xdoty;
        double wdoty;
        double wy_mult; // logically, the real w is what is in the w vector + wy_mult*Y
        matrix<double,0,1> w;
        std::vector<long> index; 
        unsigned long active_size;

        matrix<double,0,1> eig_vects_xdoty;
        matrix<double,0,1> eig_vals;
        matrix<double> eig_vects;
        matrix<double> u;

        dlib::rand rnd;


        double eps;
        unsigned long max_iterations;
        bool verbose;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_ElASTIC_NET_Hh_


