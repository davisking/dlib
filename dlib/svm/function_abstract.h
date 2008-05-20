// Copyright (C) 2007  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SVm_FUNCTION_ABSTRACT_
#ifdef DLIB_SVm_FUNCTION_ABSTRACT_

#include <cmath>
#include <limits>
#include <sstream>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename K
        >
    struct decision_function 
    {
        /*!
            REQUIREMENTS ON K
                K must be a kernel function object type as defined at the
                top of dlib/svm/kernel_abstract.h

            WHAT THIS OBJECT REPRESENTS 
                This object represents a decision or regression function that was 
                learned by a kernel based learning algorithm.  
        !*/

        typedef typename K::scalar_type scalar_type;
        typedef typename K::sample_type sample_type;
        typedef typename K::mem_manager_type mem_manager_type;

        typedef matrix<scalar_type,0,1,mem_manager_type> scalar_vector_type;
        typedef matrix<sample_type,0,1,mem_manager_type> sample_vector_type;

        const scalar_vector_type alpha;
        const scalar_type        b;
        const K                  kernel_function;
        const sample_vector_type support_vectors;

        decision_function (
        );
        /*!
            ensures
                - #b == 0
                - #alpha.nr() == 0
                - #support_vectors.nr() == 0
        !*/

        decision_function (
            const decision_function& f
        );
        /*!
            ensures
                - #*this is a copy of f
        !*/

        decision_function (
            const scalar_vector_type& alpha_,
            const scalar_type& b_,
            const K& kernel_function_,
            const sample_vector_type& support_vectors_
        ) : alpha(alpha_), b(b_), kernel_function(kernel_function_), support_vectors(support_vectors_) {}
        /*!
            ensures
                - populates the decision function with the given support vectors, weights(i.e. alphas),
                  b term, and kernel function.
        !*/

        decision_function& operator= (
            const decision_function& d
        );
        /*!
            ensures
                - #*this is identical to d
                - returns *this
        !*/

        scalar_type operator() (
            const sample_type& x
        ) const
        /*!
            ensures
                - evalutes this sample according to the decision
                  function contained in this object.
        !*/
        {
            scalar_type temp = 0;
            for (long i = 0; i < alpha.nr(); ++i)
                temp += alpha(i) * kernel_function(x,support_vectors(i));

            returns temp - b;
        }
    };

    template <
        typename K
        >
    void serialize (
        const decision_function<K>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for decision_function
    !*/

    template <
        typename K
        >
    void deserialize (
        decision_function<K>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for decision_function
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename K
        >
    struct probabilistic_decision_function 
    {
        /*!
            REQUIREMENTS ON K
                K must be a kernel function object type as defined at the
                top of dlib/svm/kernel_abstract.h

            WHAT THIS OBJECT REPRESENTS 
                This object represents a binary decision function that returns an 
                estimate of the probability that a given sample is in the +1 class.
        !*/

        typedef typename K::scalar_type scalar_type;
        typedef typename K::sample_type sample_type;
        typedef typename K::mem_manager_type mem_manager_type;

        const scalar_type a;
        const scalar_type b;
        const decision_function<K> decision_funct;

        probabilistic_decision_function (
        );
        /*!
            ensures
                - #a == 0
                - #b == 0
                - #decision_function has its initial value
        !*/

        probabilistic_decision_function (
            const probabilistic_decision_function& f
        );
        /*!
            ensures
                - #*this is a copy of f
        !*/

        probabilistic_decision_function (
            const scalar_type a_,
            const scalar_type b_,
            const decision_function<K>& decision_funct_ 
        ) : a(a_), b(b_), decision_funct(decision_funct_) {}
        /*!
            ensures
                - populates the probabilistic decision function with the given a, b, 
                  and decision_function.
        !*/

        probabilistic_decision_function& operator= (
            const probabilistic_decision_function& d
        );
        /*!
            ensures
                - #*this is identical to d
                - returns *this
        !*/

        scalar_type operator() (
            const sample_type& x
        ) const
        /*!
            ensures
                - returns a number P such that:
                    - 0 <= P <= 1
                    - P represents the probability that sample x is from 
                      the class +1
        !*/
        {
            // Evaluate the normal SVM decision function
            scalar_type f = decision_funct(x);
            // Now basically normalize the output so that it is a properly
            // conditioned probability of x being in the +1 class given
            // the output of the SVM.
            return 1/(1 + std::exp(a*f + b));
        }
    };

    template <
        typename K
        >
    void serialize (
        const probabilistic_decision_function<K>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for probabilistic_decision_function
    !*/

    template <
        typename K
        >
    void deserialize (
        probabilistic_decision_function<K>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for probabilistic_decision_function
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_FUNCTION_ABSTRACT_



