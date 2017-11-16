
#include "global_function_search.h"
#include "upper_bound_function.h"
#include "../optimization.h"

#include "../timing.h" // TODO, remove

namespace dlib
{
    namespace qopt_impl
    {
        void fit_qp_mse(
            const matrix<double>& X,
            const matrix<double,0,1>& Y,
            matrix<double>& H,
            matrix<double,0,1>& g,
            double& c
        )
        {
            DLIB_CASSERT(X.size() > 0);
            DLIB_CASSERT(X.nc() == Y.size());
            DLIB_CASSERT(X.nc() >= (X.nr()+1)*(X.nr()+2)/2);

            const long dims = X.nr();
            const long M = X.nc();

            matrix<double> W((X.nr()+1)*(X.nr()+2)/2, M);

            set_subm(W, 0,0, dims, M) = X;
            set_subm(W, dims,0, 1, M) = 1;
            for (long c = 0; c < X.nc(); ++c)
            {
                long wr = dims+1;
                for (long r = 0; r < X.nr(); ++r)
                {
                    for (long r2 = r; r2 < X.nr(); ++r2)
                    {
                        W(wr,c) = X(r,c)*X(r2,c);
                        if (r2 == r)
                            W(wr,c) *= 0.5;
                        ++wr;
                    }
                }
            }

            matrix<double,0,1> z = pinv(trans(W))*Y;

            c = z(dims);
            g = rowm(z, range(0,dims-1));

            H.set_size(dims,dims);

            long wr = dims+1;
            for (long r = 0; r < X.nr(); ++r)
            {
                for (long r2 = r; r2 < X.nr(); ++r2)
                {
                    H(r,r2) = H(r2,r) = z(wr++);
                }
            }
        }

    // ----------------------------------------------------------------------------------------

        void fit_qp(
            const matrix<double>& X,
            const matrix<double,0,1>& Y,
            matrix<double>& H,
            matrix<double,0,1>& g,
            double& c
        )
            /*!
                requires
                    - X.size() > 0
                - X.nc() == Y.size()
                - X.nr()+1 <= X.nc() <= (X.nr()+1)*(X.nr()+2)/2     
                ensures
                    - This function finds a quadratic function, Q(x), that interpolates the
                      given set of points.  If there aren't enough points to uniquely define
                  Q(x) then the Q(x) that fits the given points with the minimum Frobenius
                  norm hessian matrix is selected. 
                - To be precise:
                    - Let: Q(x) == 0.5*trans(x)*H*x + trans(x)*g + c
                    - Then this function finds H, g, and c that minimizes the following:
                        sum(squared(H))
                      such that:
                        Q(colm(X,i)) == Y(i),  for all valid i
            !*/
        {
            DLIB_CASSERT(X.size() > 0);
            DLIB_CASSERT(X.nc() == Y.size());
            DLIB_CASSERT(X.nr()+1 <= X.nc());// && X.nc() <= (X.nr()+1)*(X.nr()+2)/2);


            if (X.nc() >= (X.nr()+1)*(X.nr()+2)/2)
            {
                fit_qp_mse(X,Y,H,g,c);
                return;
            }


            const long dims = X.nr();
            const long M = X.nc();

            /*
                Our implementation uses the equations 3.9 - 3.12 from the paper:
                The NEWUOA software for unconstrained optimization without derivatives
                By M.J.D. Powell, 40th Workshop on Large Scale Nonlinear Optimization (Erice, Italy, 2004)
            */

            matrix<double> W(M + dims + 1, M + dims + 1);

            set_subm(W, 0, 0, M, M) = 0.5*squared(tmp(trans(X)*X));
            set_subm(W, 0, M, M, 1) = 1;
            set_subm(W, M, 0, 1, M) = 1;
            set_subm(W, M, M, dims+1, dims+1) = 0;
            set_subm(W, 0, M+1, X.nc(), X.nr()) = trans(X);
            set_subm(W, M+1, 0, X.nr(), X.nc()) = X;


            const matrix<double,0,1> r = join_cols(Y, zeros_matrix<double>(dims+1,1));

            //matrix<double,0,1> z = pinv(W)*r;
            lu_decomposition<decltype(W)> lu(W);
            matrix<double,0,1> z = lu.solve(r);
            if (lu.is_singular())
                std::cout << "WARNING, THE W MATRIX IS SINGULAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

            matrix<double,0,1> lambda = rowm(z, range(0,M-1));

            c = z(M);
            g = rowm(z, range(M+1,z.size()-1));
            H = X*diagm(lambda)*trans(X);
        }

    // ----------------------------------------------------------------------------------------

        struct quad_interp_result
        {
            quad_interp_result() = default;

            template <typename EXP>
            quad_interp_result(
                const matrix_exp<EXP>& best_x,
                double predicted_improvement
            ) : best_x(best_x), predicted_improvement(predicted_improvement)  {}

            matrix<double,0,1> best_x;
            double predicted_improvement = std::numeric_limits<double>::quiet_NaN();
        };

    // ----------------------------------------------------------------------------------------

        quad_interp_result find_max_quadraticly_interpolated_vector (
            const matrix<double,0,1>& anchor,
            const double radius,
            const std::vector<matrix<double,0,1>>& x,
            const std::vector<double>& y,
            const matrix<double,0,1>& lower,
            const matrix<double,0,1>& upper
        )
        {
            DLIB_CASSERT(x.size() == y.size());
            DLIB_CASSERT(x.size() > 0);
            for (size_t i = 0; i < x.size(); ++i)
                DLIB_CASSERT(anchor.size() == x[i].size());
            DLIB_CASSERT(anchor.size()+1 <= x.size() && x.size() <= (anchor.size()+1)*(anchor.size()+2)/2);


            matrix<double> X(anchor.size(), x.size());
            matrix<double,0,1> Y(x.size());
            for (size_t i = 0; i < x.size(); ++i)
            {
                set_colm(X,i) = x[i] - anchor;
                Y(i) = y[i];
            }

            matrix<double> H;
            matrix<double,0,1> g;
            double c;

            fit_qp(X, Y, H, g, c);

            matrix<double,0,1> p;

            solve_trust_region_subproblem_bounded(-H,-g, radius, p,  0.001, 500, lower-anchor, upper-anchor);

            // ensure we never move more than radius from the anchor.  This might happen if the
            // trust region subproblem isn't solved accurately enough.
            if (length(p) >= radius)
                p *= radius/length(p);


            double predicted_improvement = 0.5*trans(p)*H*p + trans(p)*g;
            return quad_interp_result{clamp(anchor+p,lower,upper), predicted_improvement};
        }

    // ----------------------------------------------------------------------------------------

        quad_interp_result pick_next_sample_quad_interp (
            const std::vector<function_evaluation>& samples,
            double& radius,
            const matrix<double,0,1>& lower,
            const matrix<double,0,1>& upper,
            const std::vector<bool>& is_integer_variable
        )
        {
            timing::block oaijsdofijas(1, "pick_next_sample_quad_interp");
            DLIB_CASSERT(samples.size() > 0);
            // We don't use the QP to optimize integer variables.  Instead, we just fix them at
            // their best observed value and use the QP to optimize the real variables.  So the
            // number of dimensions, as far as the QP is concerned, is the number of non-integer
            // variables.
            long dims = 0;
            for (auto is_int : is_integer_variable)
            {
                if (!is_int)
                    ++dims;
            }

            DLIB_CASSERT(samples.size() >= dims+1);

            // Use enough points to fill out a quadratic model or the max available if we don't
            // have quite enough.
            const long N = std::min((long)samples.size(), (dims+1)*(dims+2)/2); 


            // first find the best sample;
            double best_val = -1e300;
            matrix<double,0,1> best_x;
            for (auto& v : samples)
            {
                if (v.y > best_val)
                {
                    best_val = v.y;
                    best_x = v.x;
                }
            }

            // if there are only integer variables then there isn't really anything to do.  So just
            // return the best_x and say there is no improvement.
            if (dims == 0)
                return quad_interp_result(best_x, 0);

            matrix<long,0,1> active_dims(dims);
            long j = 0;
            for (size_t i = 0; i < is_integer_variable.size(); ++i)
            {
                if (!is_integer_variable[i])
                    active_dims(j++) = i;
            }

            // now find the N-1 nearest neighbors of best_x
            std::vector<std::pair<double,size_t>> distances;
            for (size_t i = 0; i < samples.size(); ++i)
                distances.emplace_back(length(best_x-samples[i].x), i);
            std::sort(distances.begin(), distances.end());
            distances.resize(N);

            std::vector<matrix<double,0,1>> x;
            std::vector<double> y;
            for (auto& idx : distances)
            {
                x.emplace_back(rowm(samples[idx.second].x, active_dims));
                y.emplace_back(samples[idx.second].y);
            }

            if (radius == 0)
            {
                for (auto& idx : distances)
                    radius = std::max(radius, length(rowm(best_x-samples[idx.second].x, active_dims)) );
                // Shrink the radius a little so we are always going to be making the sampling of
                // points near the best current point smaller.
                radius *= 0.95;
            }


            auto tmp = find_max_quadraticly_interpolated_vector(rowm(best_x,active_dims), radius, x, y, rowm(lower,active_dims), rowm(upper,active_dims));

            // stick the integer variables back into the solution
            for (long i = 0; i < active_dims.size(); ++i)
                best_x(active_dims(i)) = tmp.best_x(i);

            tmp.best_x = best_x;
            return tmp;
        }

    // ----------------------------------------------------------------------------------------

        matrix<double,0,1> make_random_vector(
            dlib::rand& rnd,
            const matrix<double,0,1>& lower,
            const matrix<double,0,1>& upper,
            const std::vector<bool>& is_integer_variable
        )
        {
            matrix<double,0,1> temp(lower.size());
            for (long i = 0; i < temp.size(); ++i)
            {
                temp(i) = rnd.get_double_in_range(lower(i), upper(i));
                if (is_integer_variable[i])
                    temp(i) = std::round(temp(i));
            }
            return temp;
        }

    // ----------------------------------------------------------------------------------------

        struct max_upper_bound_function 
        {
            max_upper_bound_function() = default;

            template <typename EXP>
            max_upper_bound_function(
                const matrix_exp<EXP>& x,
                double predicted_improvement,
                double upper_bound 
            ) : x(x), predicted_improvement(predicted_improvement), upper_bound(upper_bound)  {}

            matrix<double,0,1> x;
            double predicted_improvement = 0;
            double upper_bound = 0;
        };

        max_upper_bound_function pick_next_sample_max_upper_bound_function (
            dlib::rand& rnd,
            const std::vector<function_evaluation>& samples,
            const matrix<double,0,1>& lower,
            const matrix<double,0,1>& upper,
            const std::vector<bool>& is_integer_variable,
            const double relative_noise_magnitude = 0.001,
            const size_t num_random_samples = 5000
        )
        {
            timing::block oaijsdofijas(0, "pick_next_sample_max_upper_bound_function");
            DLIB_CASSERT(samples.size() > 0);
            // TODO, assert everyone has same dims


            // build the upper bound
            upper_bound_function ub(samples, relative_noise_magnitude);

            // now do a simple random search to find the maximum upper bound
            double best_ub_so_far = -std::numeric_limits<double>::infinity();
            matrix<double,0,1> vtemp(lower.size()), v;
            for (size_t rounds = 0; rounds < num_random_samples; ++rounds)
            {
                vtemp = make_random_vector(rnd, lower, upper, is_integer_variable);

                double bound = ub(vtemp);
                if (bound > best_ub_so_far)
                {
                    best_ub_so_far = bound;
                    v = vtemp;
                }
            }

            double max_value = -std::numeric_limits<double>::infinity();
            for (auto& v : samples)
                max_value = std::max(max_value, v.y);

            return max_upper_bound_function(v, best_ub_so_far - max_value, best_ub_so_far);
        }

    } // end of namespace qopt_impl;

    using namespace qopt_impl;

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    function_spec::function_spec(const matrix<double,0,1>& lower_, const matrix<double,0,1>& upper_) : lower(lower_), upper(upper_)
    {
        DLIB_CASSERT(lower.size() == upper.size());
        for (size_t i = 0; i < lower.size(); ++i)
        {
            if (upper(i) < lower(i))
                std::swap(lower(i), upper(i));
            DLIB_CASSERT(upper(i) != lower(i), "The upper and lower bounds can't be equal.");
        }
        is_integer_variable.assign(lower.size(), false);
    }

    function_spec::function_spec(const matrix<double,0,1>& lower, const matrix<double,0,1>& upper, std::vector<bool> is_integer) : function_spec(std::move(lower),std::move(upper))
    {
        is_integer_variable = std::move(is_integer);
        DLIB_CASSERT(lower.size() == (long)is_integer_variable.size());


        // Make sure any integer variables have integer bounds. 
        for (size_t i = 0; i < is_integer_variable.size(); ++i)
        {
            if (is_integer_variable[i])
            {
                DLIB_CASSERT(std::round(lower(i)) == lower(i), "If you say a variable is an integer variable then it must have an integer lower bound. \n"
                    << "lower[i] = " << lower(i));
                DLIB_CASSERT(std::round(upper(i)) == upper(i), "If you say a variable is an integer variable then it must have an integer upper bound. \n"
                    << "upper[i] = " << upper(i));
            }
        }
    }

// ----------------------------------------------------------------------------------------

    namespace gopt_impl 
    {
        std::vector<function_evaluation> funct_info::all_function_evals (
        ) const
        {
            auto temp = complete_evals;
            temp.reserve(temp.size()+incomplete_evals.size());
            // we are going to add the incomplete evals into this and assume the
            // incomplete evals are going to take y values equal to their nearest
            // neighbor complete evals.
            for (auto& eval : incomplete_evals)
                temp.emplace_back(eval.x, find_nn(complete_evals, eval.x));

            return temp;
        }

        double funct_info::find_nn (
            const std::vector<function_evaluation>& evals,
            const matrix<double,0,1>& x
        )
        {
            double best_y = 0;
            double best_dist = std::numeric_limits<double>::infinity();
            for (auto& v : evals)
            {
                double dist = length_squared(v.x-x);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_y = v.y;
                }
            }
            return best_y;
        }

    }

// ----------------------------------------------------------------------------------------

    function_evaluation_request::function_evaluation_request(function_evaluation_request&& item)
    {
        m_has_been_evaluated = item.m_has_been_evaluated;
        req = item.req;
        info = item.info;
        item.info.reset();

        item.m_has_been_evaluated = true;
    }

    function_evaluation_request& function_evaluation_request::
    operator=(function_evaluation_request&& item)
    {
        function_evaluation_request(std::move(item)).swap(*this);
        return *this;
    }

    void function_evaluation_request::
    swap(function_evaluation_request& item)
    {
        std::swap(m_has_been_evaluated, item.m_has_been_evaluated);
        std::swap(req, item.req);
        std::swap(info, item.info);
    }

    size_t function_evaluation_request::
    function_idx (
    ) const
    {
        return info->function_idx;
    }

    const matrix<double,0,1>& function_evaluation_request::
    x (
    ) const
    {
        return req.x;
    }

    bool function_evaluation_request::
    has_been_evaluated (
    ) const
    {
        return m_has_been_evaluated;
    }

    function_evaluation_request::
    ~function_evaluation_request()
    {
        if (!m_has_been_evaluated)
        {
            std::lock_guard<std::mutex> lock(*info->m);

            // remove the evaluation request from the incomplete list.
            auto i = std::find(info->incomplete_evals.begin(), info->incomplete_evals.end(), req);
            info->incomplete_evals.erase(i);
        }
    }

    void function_evaluation_request::
    set (
        double y
    )
    /*!
        requires
            - has_been_evaluated() == false
        ensures
            - #has_been_evaluated() == true
    !*/
    {
        DLIB_CASSERT(has_been_evaluated() == false);
        std::lock_guard<std::mutex> lock(*info->m);

        m_has_been_evaluated = true;


        // move the evaluation from incomplete to complete
        auto i = std::find(info->incomplete_evals.begin(), info->incomplete_evals.end(), req);
        DLIB_CASSERT(i != info->incomplete_evals.end());
        info->incomplete_evals.erase(i);
        info->complete_evals.emplace_back(req.x,y);


        // Now do trust region radius maintenance and keep track of the best objective
        // values and all that.
        if (req.was_trust_region_generated_request)
        {
            // Adjust trust region radius based on how good this evaluation
            // was.
            double measured_improvement = y-req.anchor_objective_value;
            double rho = measured_improvement/std::abs(req.predicted_improvement);
            if (rho < 0.25)
                info->radius *= 0.5;
            else if (rho > 0.75)
                info->radius *= 2;
        }

        if (y > info->best_objective_value)
        {
            if (length(req.x - info->best_x) > info->radius*1.001)
            {
                // reset trust region radius since we made a big move.  Doing this will
                // cause the radius to be reset to the size of the local region.
                info->radius = 0;
            }
            info->best_objective_value = y;
            info->best_x = std::move(req.x);
        }
    }

// ----------------------------------------------------------------------------------------

    global_function_search::
    global_function_search(
        const function_spec& function
    ) : global_function_search(std::vector<function_spec>(1,function)) {}

    global_function_search::
    global_function_search(
        const std::vector<function_spec>& functions_
    )
    {
        m = std::make_shared<std::mutex>();
        functions.reserve(functions_.size());
        for (size_t i = 0; i < functions_.size(); ++i)
            functions.emplace_back(std::make_shared<gopt_impl::funct_info>(functions_[i],i,m));
    }

    global_function_search::
    global_function_search(
        const std::vector<function_spec>& functions_,
        const std::vector<std::vector<function_evaluation>>& initial_function_evals
    ) : global_function_search(functions_)
    {
        DLIB_CASSERT(functions_.size() == initial_function_evals.size());
        for (size_t i = 0; i < initial_function_evals.size(); ++i)
        {
            functions[i]->complete_evals = initial_function_evals[i];
        }
    }


    size_t global_function_search::
    num_functions() const 
    { 
        return functions.size();
    }

    void global_function_search::
    set_seed (
        time_t seed
    )
    {
        rnd = dlib::rand(seed);
    }

    void global_function_search::
    get_function_evaluations (
        std::vector<function_spec>& specs,
        std::vector<std::vector<function_evaluation>>& function_evals
    ) const
    {
        std::lock_guard<std::mutex> lock(*m);
        specs.clear();
        function_evals.clear();
        for (size_t i = 0; i < functions.size(); ++i)
        {
            specs.emplace_back(functions[i]->spec);
            function_evals.emplace_back(functions[i]->complete_evals);
        }
    }

    void global_function_search::
    get_best_function_eval (
        matrix<double,0,1>& x,
        double& y,
        size_t& function_idx
    ) const
    {
        DLIB_CASSERT(num_functions() != 0);

        std::lock_guard<std::mutex> lock(*m);

        // find the largest value
        auto& info = *best_function(function_idx);
        y = info.best_objective_value;
        x = info.best_x;
    }

    function_evaluation_request global_function_search::
    get_next_x (
    ) 
    {
        DLIB_CASSERT(num_functions() != 0);

        using namespace gopt_impl;

        std::lock_guard<std::mutex> lock(*m);


        // the first thing we do is make sure each function has at least max(3,dimensionality of function) evaluations
        for (auto& info : functions)
        {
            const long dims = info->spec.lower.size();
            if (info->complete_evals.size() < std::max<long>(3,dims))
            {
                outstanding_function_eval_request new_req;
                new_req.request_id = next_request_id++;
                new_req.x = make_random_vector(rnd, info->spec.lower, info->spec.upper, info->spec.is_integer_variable);
                info->incomplete_evals.emplace_back(new_req);
                return function_evaluation_request(new_req,info);
            }
        }


        if (do_trust_region_step && !has_incomplete_trust_region_request())
        {
            // find the currently best performing function, we will do a trust region
            // step on it.
            auto info = best_function();
            const long dims = info->spec.lower.size();
            // if we have enough points to do a trust region step
            if (info->complete_evals.size() > dims+1)
            {
                auto tmp = pick_next_sample_quad_interp(info->complete_evals,
                    info->radius, info->spec.lower, info->spec.upper, info->spec.is_integer_variable);
                if (tmp.predicted_improvement > qp_eps)
                {
                    do_trust_region_step = false;
                    outstanding_function_eval_request new_req;
                    new_req.request_id = next_request_id++;
                    new_req.x = tmp.best_x;
                    new_req.was_trust_region_generated_request = true;
                    new_req.anchor_objective_value = info->best_objective_value;
                    new_req.predicted_improvement = tmp.predicted_improvement;
                    info->incomplete_evals.emplace_back(new_req);
                    return function_evaluation_request(new_req, info);
                }
            }
        }

        // make it so we alternate between upper bounded and trust region steps.
        do_trust_region_step = true;

        if (rnd.get_random_double() >= pure_random_search_probability)
        {
            // pick a point at random to sample according to the upper bound
            double best_upper_bound = -std::numeric_limits<double>::infinity();
            std::shared_ptr<funct_info> best_funct;
            matrix<double,0,1> next_sample;
            // so figure out if any function has a good upper bound and if so pick the
            // function with the largest upper bound for evaluation.
            for (auto& info : functions)
            {
                auto tmp = pick_next_sample_max_upper_bound_function(rnd,
                    info->all_function_evals(), info->spec.lower, info->spec.upper,
                    info->spec.is_integer_variable,  relative_noise_magnitude, num_random_samples);
                if (tmp.predicted_improvement > 0 && tmp.upper_bound > best_upper_bound) 
                {
                    best_upper_bound = tmp.upper_bound;
                    next_sample = std::move(tmp.x);
                    best_funct = info;
                }
            }

            // if we found a good function to evaluate then return that. 
            if (best_funct)
            {
                outstanding_function_eval_request new_req;
                new_req.request_id = next_request_id++;
                new_req.x = std::move(next_sample);
                best_funct->incomplete_evals.emplace_back(new_req);
                return function_evaluation_request(new_req, best_funct);
            }
        }


        // pick entirely at random
        size_t function_idx = rnd.get_integer(functions.size());
        auto info = functions[function_idx];
        outstanding_function_eval_request new_req;
        new_req.request_id = next_request_id++;
        new_req.x = make_random_vector(rnd, info->spec.lower, info->spec.upper, info->spec.is_integer_variable);
        info->incomplete_evals.emplace_back(new_req);
        return function_evaluation_request(new_req, info);

    }

    double global_function_search::
    get_pure_random_search_probability (
    ) const { return pure_random_search_probability; }

    void global_function_search::
    set_pure_random_search_probability (
        double prob
    ) 
    {
        DLIB_CASSERT(0 <= prob && prob <= 1);
        pure_random_search_probability = prob;
    }

    double global_function_search::
    get_solver_epsilon (
    ) const { return qp_eps; }

    void global_function_search::
    set_solver_epsilon (
        double eps
    )
    {
        DLIB_CASSERT(0 <= eps);
        qp_eps = eps;
    }

    double global_function_search::
    get_relative_noise_magnitude (
    ) const { return relative_noise_magnitude; }
    void global_function_search::
    set_relative_noise_magnitude (
        double value
    )
    {
        DLIB_CASSERT(0 <= value);
        relative_noise_magnitude = value;
    }

    size_t global_function_search::
    get_monte_carlo_upper_bound_sample_num (
    ) const { return num_random_samples; }
    void global_function_search::
    set_monte_carlo_upper_bound_sample_num (
        size_t num
    )
    {
        DLIB_CASSERT(0 <= num);
        num_random_samples = num;
    }


    std::shared_ptr<gopt_impl::funct_info> global_function_search::
    best_function() const
    {
        size_t idx = 0;
        return best_function(idx);
    }

    std::shared_ptr<gopt_impl::funct_info> global_function_search::
    best_function(size_t& idx) const
    {
        auto i = std::max_element(functions.begin(), functions.end(), 
            [](const std::shared_ptr<gopt_impl::funct_info>& a, const std::shared_ptr<gopt_impl::funct_info>& b) { return a->best_objective_value < b->best_objective_value; });

        idx = std::distance(functions.begin(),i);
        return *i;
    }

    bool global_function_search::
    has_incomplete_trust_region_request (
    ) const 
    {
        for (auto& f : functions)
        {
            for (auto& i : f->incomplete_evals)
            {
                if (i.was_trust_region_generated_request)
                    return true;
            }
        }
        return false;
    }

// ----------------------------------------------------------------------------------------

}

