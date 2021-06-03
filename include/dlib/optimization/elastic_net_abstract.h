// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_ElASTIC_NET_ABSTRACT_Hh_
#ifdef DLIB_ElASTIC_NET_ABSTRACT_Hh_

#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class elastic_net
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for solving the following optimization problem:

                    min_w:      length_squared(X*w - Y) + ridge_lambda*length_squared(w)
                    such that:  sum(abs(w)) <= lasso_budget

                That is, it solves the elastic net optimization problem.  This object also
                has the special property that you can quickly obtain different solutions
                for different settings of ridge_lambda, lasso_budget, and target Y values.

                This is because a large amount of work is precomputed in the constructor.
                The solver will also remember the previous solution and will use that to
                warm start subsequent invocations.  Therefore, you can efficiently get
                solutions for a wide range of regularization parameters.
                
                
                The particular algorithm used to solve it is described in the paper:
                    Zhou, Quan, et al. "A reduction of the elastic net to support vector
                    machines with an application to gpu computing." arXiv preprint
                    arXiv:1409.1976 (2014).  APA 

                And for the SVM solver sub-component we use the algorithm from:
                    Hsieh, Cho-Jui, et al. "A dual coordinate descent method for large-scale
                    linear SVM." Proceedings of the 25th international conference on Machine
                    learning. ACM, 2008. 
        !*/

    public:

        template <typename EXP>
        explicit elastic_net(
            const matrix_exp<EXP>& XX
        ); 
        /*!
            requires
                - XX.size() != 0
                - XX.nr() == XX.nc()
            ensures
                - #get_epsilon() == 1e-5
                - #get_max_iterations() == 50000
                - This object will not be verbose unless be_verbose() is called.
                - #size() == XX.nc()
                - #have_target_values() == false
                - We interpret XX as trans(X)*X where X is as defined in the objective
                  function discussed above in WHAT THIS OBJECT REPRESENTS.
        !*/

        template <typename EXP1, typename EXP2>
        elastic_net(
            const matrix_exp<EXP1>& XX,
            const matrix_exp<EXP2>& XY
        ); 
        /*!
            requires
                - XX.size() != 0
                - XX.nr() == XX.nc()
                - is_col_vector(XY)
                - XX.nc() == Y.size()
            ensures
                - constructs this object by calling the elastic_net(XX) constructor and
                  then calling this->set_xy(XY).
                - #have_target_values() == true 
                - We interpret XX as trans(X)*X where X is as defined in the objective
                  function discussed above in WHAT THIS OBJECT REPRESENTS.  Similarly, XY
                  should be trans(X)*Y.
        !*/

        long size (
        ) const; 
        /*!
            ensures
                - returns the dimensionality of the data loaded into this object.  That is,
                  how many elements are in the optimal w vector?  This function returns
                  that number.
        !*/

        bool have_target_values (
        ) const;
        /*!
            ensures
                - returns true if set_xy() has been called and false otherwise.
        !*/

        template <typename EXP>
        void set_xy(
            const matrix_exp<EXP>& XY
        );
        /*!
            requires
                - is_col_vector(Y)
                - Y.size() == size()
            ensures
                - #have_target_values() == true
                - Sets the target values of the regression.  Note that we expect the given
                  matrix, XY, to be equal to trans(X)*Y, where X and Y have the definitions
                  discussed above in WHAT THIS OBJECT REPRESENTS.
        !*/

        void set_epsilon(
            double eps
        );
        /*!
            requires
                - eps > 0
            ensures
                - #get_epsilon() == eps
        !*/

        double get_epsilon (
        ) const;
        /*!
            ensures
                - returns the error epsilon that determines when the solver should stop.
                  Smaller values may result in a more accurate solution but take longer to
                  execute.  
        !*/

        unsigned long get_max_iterations (
        ) const; 
        /*!
            ensures
                - returns the maximum number of iterations the optimizer is allowed to run
                  before it is required to stop and return a result.
        !*/

        void set_max_iterations (
            unsigned long max_iter
        );
        /*!
            ensures
                - #get_max_iterations() == max_iter
        !*/

        void be_verbose (
        );
        /*!
            ensures
                - This object will print status messages to standard out so that a 
                  user can observe the progress of the algorithm.
        !*/

        void be_quiet (
        );
        /*!
            ensures
                - this object will not print anything to standard out.
        !*/


        matrix<double,0,1> operator() (
            double ridge_lambda,
            double lasso_budget = std::numeric_limits<double>::infinity()
        );
        /*!
            requires
                - have_target_values() == true
                - ridge_lambda > 0
                - lasso_budget > 0
            ensures
                - Solves the optimization problem described in the WHAT THIS OBJECT
                  REPRESENTS section above and returns the optimal w.
                - The returned vector has size() elements.
                - if (lasso_budget == infinity) then
                    - The lasso constraint is ignored 
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_ElASTIC_NET_ABSTRACT_Hh_


