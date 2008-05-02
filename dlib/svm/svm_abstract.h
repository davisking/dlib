// Copyright (C) 2007  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SVm_ABSTRACT_
#ifdef DLIB_SVm_ABSTRACT_

#include <cmath>
#include <limits>
#include <sstream>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
/*!A                               Kernel_Function_Objects                               */
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    /*! 
        WHAT IS A KERNEL FUNCTION OBJECT?
            In the context of the dlib library documentation a kernel function object
            is an object with an interface with the following properties:
                - a public typedef named sample_type
                - a public typedef named scalar_type which should be a float, double, or 
                  long double type.
                - an overloaded operator() that operates on two items of sample_type 
                  and returns a scalar_type.  
                  (e.g. scalar_type val = kernel_function(sample(i),sample(j)); 
                   would be a valid expression)
                - a public typedef named mem_manager_type that is an implementation of 
                  dlib/memory_manager/memory_manager_kernel_abstract.h or
                  dlib/memory_manager_global/memory_manager_global_kernel_abstract.h or
                  dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h 

        For examples of kernel functions see the following objects
        (e.g. the radial_basis_kernel).
    !*/

    template <
        typename T
        >
    struct radial_basis_kernel
    {
        /*!
            REQUIREMENTS ON T
                T must be a dlib::matrix object 

            WHAT THIS OBJECT REPRESENTS
                This object represents a radial basis function kernel
        !*/

        typedef typename T::type scalar_type;
        typedef T sample_type;
        typedef typename T::mem_manager_type mem_manager_type;

        const scalar_type gamma;

        radial_basis_kernel(
        );
        /*!
            ensures
                - #gamma == 0.1 
        !*/

        radial_basis_kernel(
            const radial_basis_kernel& k
        );
        /*!
            ensures
                - #gamma == k.gamma
        !*/

        radial_basis_kernel(
            const scalar_type g
        );
        /*!
            ensures
                - #gamma == g
        !*/

        scalar_type operator() (
            const sample_type& a,
            const sample_type& b
        ) const;
        /*!
            requires
                - a.nc() == 1
                - b.nc() == 1
                - a.nr() == b.nr()
            ensures
                - returns exp(-gamma * ||a-b||^2)
        !*/

        radial_basis_kernel& operator= (
            const radial_basis_kernel& k
        );
        /*!
            ensures
                - #gamma = k.gamma
                - returns *this
        !*/

    };

    template <
        typename T
        >
    void serialize (
        const radial_basis_kernel<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for radial_basis_kernel
    !*/

    template <
        typename K
        >
    void deserialize (
        radial_basis_kernel<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support for radial_basis_kernel
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    struct polynomial_kernel
    {
        /*!
            REQUIREMENTS ON T
                T must be a dlib::matrix object 

            WHAT THIS OBJECT REPRESENTS
                This object represents a polynomial kernel
        !*/

        typedef typename T::type scalar_type;
        typedef T sample_type;
        typedef typename T::mem_manager_type mem_manager_type;

        const scalar_type gamma;
        const scalar_type coef;
        const scalar_type degree;

        polynomial_kernel(
        );
        /*!
            ensures
                - #gamma == 1 
                - #coef == 0 
                - #degree == 1 
        !*/

        polynomial_kernel(
            const radial_basis_kernel& k
        );
        /*!
            ensures
                - #gamma == k.gamma
        !*/

        polynomial_kernel(
            const scalar_type g,
            const scalar_type c,
            const scalar_type d
        );
        /*!
            ensures
                - #gamma == g
                - #coef == c
                - #degree == d
        !*/

        scalar_type operator() (
            const sample_type& a,
            const sample_type& b
        ) const;
        /*!
            requires
                - a.nc() == 1
                - b.nc() == 1
                - a.nr() == b.nr()
            ensures
                - returns pow(gamma*trans(a)*b + coef, degree)
        !*/

        polynomial_kernel& operator= (
            const polynomial_kernel& k
        );
        /*!
            ensures
                - #gamma = k.gamma
                - #coef = k.coef
                - #degree = k.degree
                - returns *this
        !*/

    };

    template <
        typename T
        >
    void serialize (
        const polynomial_kernel<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for polynomial_kernel
    !*/

    template <
        typename K
        >
    void deserialize (
        polynomial_kernel<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support for polynomial_kernel
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    struct linear_kernel
    {
        /*!
            REQUIREMENTS ON T
                T must be a dlib::matrix object 

            WHAT THIS OBJECT REPRESENTS
                This object represents a linear function kernel
        !*/

        typedef typename T::type scalar_type;
        typedef T sample_type;
        typedef typename T::mem_manager_type mem_manager_type;

        scalar_type operator() (
            const sample_type& a,
            const sample_type& b
        ) const;
        /*!
            requires
                - a.nc() == 1
                - b.nc() == 1
                - a.nr() == b.nr()
            ensures
                - returns trans(a)*b
        !*/
    };

    template <
        typename T
        >
    void serialize (
        const linear_kernel<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for linear_kernel
    !*/

    template <
        typename K
        >
    void deserialize (
        linear_kernel<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support for linear_kernel 
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename K
        >
    struct decision_function 
    {
        /*!
            REQUIREMENTS ON K
                K must be a kernel function object type as defined at the top 
                of this document.

            WHAT THIS OBJECT REPRESENTS 
                This object represents a binary decision function.
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
                - predicts the class of x.  
                - if (the class is predicted to be +1) then
                    - returns a number >= 0
                - else
                    - returns a number < 0
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
                K must be a kernel function object type as defined at the top 
                of this document.

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
// ----------------------------------------------------------------------------------------
//                              Functions that perform SVM training 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    typename T::type maximum_nu (
        const T& y
    );
    /*!
        requires
            - T == a matrix object 
            - y.nc() == 1
            - y.nr() > 1
            - for all valid i:
                - y(i) == -1 or +1
        ensures
            - returns the maximum valid nu that can be used with svm_nu_train().
              (i.e. 2.0*min(number of +1 examples in y, number of -1 examples in y)/y.nr())
    !*/

    template <
        typename K
        >
    const decision_function<K> svm_nu_train (
        const typename decision_function<K>::sample_vector_type& x,
        const typename decision_function<K>::scalar_vector_type& y,
        const K&                       kernel_function,
        const typename K::scalar_type  nu,
        const long                     cache_size = 200,
        const typename K::scalar_type  eps = 0.001
    );
    /*!
        requires
            - eps > 0
            - x.nc() == 1 (i.e. x is a column vector)
            - y.nc() == 1 (i.e. y is a column vector)
            - x.nr() == y.nr() 
            - x.nr() > 1
            - cache_size > 0
            - for all valid i:
                - y(i) == -1 or +1
                - y(i) is the class that should be assigned to training example x(i)
            - 0 < nu < maximum_nu(y) 
            - kernel_function == a kernel function object type as defined at the top 
              of this document.
        ensures
            - trains a nu support vector classifier given the training samples in x and 
              labels in y.  Training is done when the error is less than eps.
            - caches approximately at most cache_size megabytes of the kernel matrix. 
              (bigger values of this may make training go faster but doesn't affect the 
              result.  However, too big a value will cause you to run out of memory.)
            - returns the resulting decision function 
    !*/

    /*
        The implementation of the nu-svm training algorithm used by this library is based
        on the following excellent papers:
            - Chang and Lin, Training {nu}-Support Vector Classifiers: Theory and Algorithms
            - Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector 
              machines, 2001. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm
    */

// ----------------------------------------------------------------------------------------

    template <
        typename K
        >
    const probabilistic_decision_function<K> svm_nu_train_prob (
        const typename decision_function<K>::sample_vector_type& x,
        const typename decision_function<K>::scalar_vector_type& y,
        const K&                       kernel_function,
        const typename K::scalar_type  nu,
        const long                     folds,
        const long                     cache_size = 200,
        const typename K::scalar_type  eps = 0.001
    );
    /*!
        requires
            - eps > 0
            - 1 < folds <= x.nr()
            - x.nc() == 1 (i.e. x is a column vector)
            - y.nc() == 1 (i.e. y is a column vector)
            - x.nr() == y.nr() 
            - x.nr() > 1
            - cache_size > 0
            - for all valid i:
                - y(i) == -1 or +1
                - y(i) is the class that should be assigned to training example x(i)
            - 0 < nu < maximum_nu(y) 
            - kernel_function == a kernel function object type as defined at the top 
              of this document.
        ensures
            - trains a nu support vector classifier given the training samples in x and 
              labels in y.  Training is done when the error is less than eps.
            - caches approximately at most cache_size megabytes of the kernel matrix. 
              (bigger values of this may make training go faster but doesn't affect the 
              result.  However, too big a value will cause you to run out of memory.)
            - returns a probabilistic_decision_function that represents the trained
              svm.
            - The parameters of the probability model are estimated by performing k-fold 
              cross validation. 
            - The number of folds used is given by the folds argument.
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                                  Miscellaneous functions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename K
        >
    const matrix<typename K::scalar_type, 1, 2, typename K::mem_manager_type> svm_nu_cross_validate (
        const typename decision_function<K>::sample_vector_type& x,
        const typename decision_function<K>::scalar_vector_type& y,
        const K&                       kernel_function,
        const typename K::scalar_type  nu,
        const long                     folds,
        const long                     cache_size = 200,
        const typename K::scalar_type  eps = 0.001
    );
    /*!
        requires
            - eps > 0
            - 1 < folds <= x.nr()
            - x.nc() == 1 (i.e. x is a column vector)
            - y.nc() == 1 (i.e. y is a column vector)
            - x.nr() == y.nr() 
            - x.nr() > 1
            - cache_size > 0
            - for all valid i:
                - y(i) == -1 or +1
                - y(i) is the class that should be assigned to training example x(i)
            - 0 < nu < maximum_nu(y) 
            - kernel_function == a kernel function object type as defined at the top 
              of this document.
        ensures
            - performs k-fold cross validation by training a nu-svm using the svm_nu_train()
              function.  Each fold is tested using the learned decision_function and the
              average accuracy from all folds is returned.  The accuracy is returned in
              a column vector, let us call it R.  Both quantities in R are numbers between
              0 and 1 which represent the fraction of examples correctly classified.  R(0) 
              is the fraction of +1 examples correctly classified and R(1) is the fraction
              of -1 examples correctly classified.
            - The number of folds used is given by the folds argument.
            - in each fold: trains a nu support vector classifier given the training samples 
              in x and labels in y.  Training is done when the error is less than eps.
            - caches approximately at most cache_size megabytes of the kernel matrix. 
              (bigger values of this may make training go faster but doesn't affect the 
              result.  However, too big a value will cause you to run out of memory.)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        typename U
        >
    void randomize_samples (
        T& samples,
        U& labels 
    );
    /*!
        requires
            - T == a matrix object that contains a swappable type
            - U == a matrix object that contains a swappable type
            - samples.nc() == 1
            - labels.nc() == 1
            - samples.nr() == labels.nr()
        ensures
            - randomizes the order of the samples and labels but preserves
              the pairing between each sample and its label
            - for all valid i:
                - let r == the random index samples(i) was moved to.  then:
                    - #labels(r) == labels(i)
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_ABSTRACT_


