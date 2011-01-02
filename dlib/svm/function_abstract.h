// Copyright (C) 2007  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SVm_FUNCTION_ABSTRACT_
#ifdef DLIB_SVm_FUNCTION_ABSTRACT_

#include <cmath>
#include <limits>
#include <sstream>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "../serialize.h"
#include "../statistics/statistics_abstract.h"

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
                This object represents a classification or regression function that was 
                learned by a kernel based learning algorithm.   Therefore, it is a function 
                object that takes a sample object and returns a scalar value.
        !*/

        typedef K kernel_type;
        typedef typename K::scalar_type scalar_type;
        typedef typename K::sample_type sample_type;
        typedef typename K::mem_manager_type mem_manager_type;

        typedef matrix<scalar_type,0,1,mem_manager_type> scalar_vector_type;
        typedef matrix<sample_type,0,1,mem_manager_type> sample_vector_type;

        scalar_vector_type alpha;
        scalar_type        b;
        K                  kernel_function;
        sample_vector_type basis_vectors;

        decision_function (
        );
        /*!
            ensures
                - #b == 0
                - #alpha.nr() == 0
                - #basis_vectors.nr() == 0
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
            const sample_vector_type& basis_vectors_
        ) : alpha(alpha_), b(b_), kernel_function(kernel_function_), basis_vectors(basis_vectors_) {}
        /*!
            ensures
                - populates the decision function with the given basis vectors, weights(i.e. alphas),
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
                - evaluates this sample according to the decision
                  function contained in this object.
        !*/
        {
            scalar_type temp = 0;
            for (long i = 0; i < alpha.nr(); ++i)
                temp += alpha(i) * kernel_function(x,basis_vectors(i));

            return temp - b;
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
        typename function_type 
        >
    struct probabilistic_function 
    {
        /*!
            REQUIREMENTS ON function_type 
                - function_type must be a function object with an overloaded
                  operator() similar to the other function objects defined in
                  this file.  The operator() should return a scalar type such as
                  double or float.

            WHAT THIS OBJECT REPRESENTS 
                This object represents a binary decision function that returns an 
                estimate of the probability that a given sample is in the +1 class.
        !*/

        typedef typename function_type::scalar_type scalar_type;
        typedef typename function_type::sample_type sample_type;
        typedef typename function_type::mem_manager_type mem_manager_type;

        scalar_type alpha;
        scalar_type beta;
        function_type decision_funct;

        probabilistic_function (
        );
        /*!
            ensures
                - #alpha == 0
                - #beta == 0
                - #decision_funct has its initial value
        !*/

        probabilistic_function (
            const probabilistic_function& f
        );
        /*!
            ensures
                - #*this is a copy of f
        !*/

        probabilistic_function (
            const scalar_type a,
            const scalar_type b,
            const function_type& decision_funct_ 
        ) : alpha(a), beta(b), decision_funct(decision_funct_) {}
        /*!
            ensures
                - populates the probabilistic decision function with the given alpha, beta, 
                  and decision function.
        !*/

        probabilistic_function& operator= (
            const probabilistic_function& d
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
            // Evaluate the normal decision function
            scalar_type f = decision_funct(x);
            // Now basically normalize the output so that it is a properly
            // conditioned probability of x being in the +1 class given
            // the output of the decision function.
            return 1/(1 + std::exp(alpha*f + beta));
        }
    };

    template <
        typename function_type
        >
    void serialize (
        const probabilistic_function<function_type>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for probabilistic_function
    !*/

    template <
        typename function_type
        >
    void deserialize (
        probabilistic_function<function_type>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for probabilistic_function
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

                Note that this object is essentially just a copy of 
                probabilistic_function but with the template argument 
                changed from being a function type to a kernel type.  Therefore, this
                type is just a convenient version of probabilistic_function
                for the case where the decision function is a dlib::decision_function<K>.
        !*/

        typedef K kernel_type;
        typedef typename K::scalar_type scalar_type;
        typedef typename K::sample_type sample_type;
        typedef typename K::mem_manager_type mem_manager_type;

        scalar_type alpha;
        scalar_type beta;
        decision_function<K> decision_funct;

        probabilistic_decision_function (
        );
        /*!
            ensures
                - #alpha == 0
                - #beta == 0
                - #decision_funct has its initial value
        !*/

        probabilistic_decision_function (
            const probabilistic_decision_function& f
        );
        /*!
            ensures
                - #*this is a copy of f
        !*/

        probabilistic_decision_function (
            const probabilistic_function<decision_function<K> >& d
        );
        /*!
            ensures
                - #*this is a copy of f
        !*/

        probabilistic_decision_function (
            const scalar_type a,
            const scalar_type b,
            const decision_function<K>& decision_funct_ 
        ) : alpha(a), beta(b), decision_funct(decision_funct_) {}
        /*!
            ensures
                - populates the probabilistic decision function with the given alpha, beta, 
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
            // Evaluate the normal decision function
            scalar_type f = decision_funct(x);
            // Now basically normalize the output so that it is a properly
            // conditioned probability of x being in the +1 class given
            // the output of the decision function.
            return 1/(1 + std::exp(alpha*f + beta));
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

    template <
        typename K
        >
    struct distance_function 
    {
        /*!
            REQUIREMENTS ON K
                K must be a kernel function object type as defined at the
                top of dlib/svm/kernel_abstract.h

            WHAT THIS OBJECT REPRESENTS 
                This object represents a point in kernel induced feature space. 
                You may use this object to find the distance from the point it 
                represents to points in input space as well as other points
                represented by distance_functions.

                Any routine that creates a distance_function should always
                automatically populate the this->b field.  But for reference, 
                this->b is supposed to contain the squared norm of the point
                in kernel feature space.  So this means that if this function
                is to compute a proper distance then this->b should always be equal 
                to the following:
                    trans(alpha)*kernel_matrix(kernel_function,basis_vectors)*alpha
        !*/

        typedef K kernel_type;
        typedef typename K::scalar_type scalar_type;
        typedef typename K::sample_type sample_type;
        typedef typename K::mem_manager_type mem_manager_type;

        typedef matrix<scalar_type,0,1,mem_manager_type> scalar_vector_type;
        typedef matrix<sample_type,0,1,mem_manager_type> sample_vector_type;

        scalar_vector_type alpha;
        scalar_type        b;
        K                  kernel_function;
        sample_vector_type basis_vectors;

        distance_function (
        );
        /*!
            ensures
                - #b == 0
                - #alpha.nr() == 0
                - #basis_vectors.nr() == 0
        !*/

        distance_function (
            const distance_function& f
        );
        /*!
            ensures
                - #*this is a copy of f
        !*/

        distance_function (
            const scalar_vector_type& alpha_,
            const scalar_type& b_,
            const K& kernel_function_,
            const sample_vector_type& basis_vectors_
        ) : alpha(alpha_), b(b_), kernel_function(kernel_function_), basis_vectors(basis_vectors_) {}
        /*!
            ensures
                - populates the distance function with the given basis vectors, weights(i.e. alphas),
                  b term, and kernel function.
        !*/

        distance_function& operator= (
            const distance_function& d
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
                - Let O(x) represent the point x projected into kernel induced feature space.
                - let c == sum alpha(i)*O(basis_vectors(i)) == the point in kernel space that
                  this object represents.
                - Then this object returns the distance between the point O(x) and c in kernel
                  space. 
        !*/
        {
            scalar_type temp = 0;
            for (long i = 0; i < alpha.nr(); ++i)
                temp += alpha(i) * kernel_function(x,basis_vectors(i));

            temp = b + kernel_function(x,x) - 2*temp; 
            if (temp > 0)
                return std::sqrt(temp);
            else
                return 0;
        }

        scalar_type operator() (
            const distance_function& x
        ) const
        /*!
            ensures
                - returns the distance between the points in kernel space represented by *this and x.
        !*/
        {
            scalar_type temp = 0;
            for (long i = 0; i < alpha.nr(); ++i)
                for (long j = 0; j < x.alpha.nr(); ++j)
                    temp += alpha(i)*x.alpha(j) * kernel_function(basis_vectors(i), x.basis_vectors(j));

            temp = b + x.b - 2*temp;
            if (temp > 0)
                return std::sqrt(temp);
            else
                return 0;
        }
    };

    template <
        typename K
        >
    void serialize (
        const distance_function<K>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for distance_function
    !*/

    template <
        typename K
        >
    void deserialize (
        distance_function<K>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for distance_function
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename function_type,
        typename normalizer_type = vector_normalizer<typename function_type::sample_type>
        >
    struct normalized_function 
    {
        /*!
            REQUIREMENTS ON function_type 
                - function_type must be a function object with an overloaded
                  operator() similar to the other function objects defined in
                  this file.

            REQUIREMENTS ON normalizer_type
                - normalizer_type must be a function object with an overloaded
                  operator() that takes a sample_type and returns a sample_type.

            WHAT THIS OBJECT REPRESENTS 
                This object represents a container for another function
                object and an instance of a normalizer function.  

                It automatically normalizes all inputs before passing them
                off to the contained function object.
        !*/

        typedef typename function_type::scalar_type scalar_type;
        typedef typename function_type::sample_type sample_type;
        typedef typename function_type::mem_manager_type mem_manager_type;

        normalizer_type normalizer;
        function_type function;

        normalized_function (
        );
        /*!
            ensures
                - the members of this object have their default values
        !*/

        normalized_function (
            const normalized_function& f
        );
        /*!
            ensures
                - #*this is a copy of f
        !*/

        normalized_function (
            const vector_normalizer<sample_type>& normalizer_,
            const function_type& funct 
        ) : normalizer(normalizer_), function(funct) {}
        /*!
            ensures
                - populates this object with the vector_normalizer and function object 
        !*/

        normalized_function& operator= (
            const normalized_function& d
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
                - returns function(normalizer(x))
        !*/
    };

    template <
        typename function_type,
        typename normalizer_type 
        >
    void serialize (
        const normalized_function<function_type, normalizer_type>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for normalized_function
    !*/

    template <
        typename function_type,
        typename normalizer_type 
        >
    void deserialize (
        normalized_function<function_type, normalizer_type>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for normalized_function
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename K
        >
    struct projection_function 
    {
        /*!
            REQUIREMENTS ON K
                K must be a kernel function object type as defined at the
                top of dlib/svm/kernel_abstract.h

            WHAT THIS OBJECT REPRESENTS 
                This object represents a function that takes a data sample and projects
                it into kernel feature space.  The result is a real valued column vector that 
                represents a point in a kernel feature space.
        !*/

        typedef K kernel_type;
        typedef typename K::scalar_type scalar_type;
        typedef typename K::sample_type sample_type;
        typedef typename K::mem_manager_type mem_manager_type;

        typedef matrix<scalar_type,0,1,mem_manager_type> scalar_vector_type;
        typedef matrix<scalar_type,0,0,mem_manager_type> scalar_matrix_type;
        typedef matrix<sample_type,0,1,mem_manager_type> sample_vector_type;

        scalar_matrix_type weights;
        K                  kernel_function;
        sample_vector_type basis_vectors;

        projection_function (
        );
        /*!
            ensures
                - #weights.size() == 0
                - #basis_vectors.size() == 0
        !*/

        projection_function (
            const projection_function& f
        );
        /*!
            ensures
                - #*this is a copy of f
        !*/

        projection_function (
            const scalar_matrix_type& weights_,
            const K& kernel_function_,
            const sample_vector_type& basis_vectors_
        ) : weights(weights_), kernel_function(kernel_function_), basis_vectors(basis_vectors_) {}
        /*!
            ensures
                - populates the projection function with the given basis vectors, weights,
                  and kernel function.
        !*/

        projection_function& operator= (
            const projection_function& d
        );
        /*!
            ensures
                - #*this is identical to d
                - returns *this
        !*/

        long out_vector_size (
        ) const;
        /*!
            ensures
                - returns weights.nr()
                  (i.e. returns the dimensionality of the vectors output by this projection_function.)
        !*/

        const scalar_vector_type& operator() (
            const sample_type& x
        ) const
        /*!
            requires
                - weights.nc() == basis_vectors.size()
                - out_vector_size() > 0
            ensures
                - Takes the given x sample and projects it onto part of the kernel feature 
                  space spanned by the basis_vectors.  The exact projection arithmetic is 
                  defined below.
        !*/
        {
            // Run the x sample through all the basis functions we have and then
            // multiply it by the weights matrix and return the result.  Note that
            // the temp vectors are here to avoid reallocating their memory every
            // time this function is called.
            temp1 = kernel_matrix(kernel_function, basis_vectors, x);
            temp2 = weights*temp1;
            return temp2;
        }

    private:
        mutable scalar_vector_type temp1, temp2;
    };

    template <
        typename K
        >
    void serialize (
        const projection_function<K>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for projection_function
    !*/

    template <
        typename K
        >
    void deserialize (
        projection_function<K>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for projection_function
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_FUNCTION_ABSTRACT_



