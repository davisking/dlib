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
#include "function_abstract.h"
#include "kernel_abstract.h"

namespace dlib
{

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
            - kernel_function == a kernel function object type as defined at the
              top of dlib/svm/kernel_abstract.h
        ensures
            - trains a nu support vector classifier given the training samples in x and 
              labels in y.  Training is done when the error is less than eps.
            - caches approximately at most cache_size megabytes of the kernel matrix. 
              (bigger values of this may make training go faster but doesn't affect the 
              result.  However, too big a value will cause you to run out of memory.)
            - returns a decision function F with the following properties:
                - if (new_x is a sample predicted have +1 label) then
                    - F(new_x) >= 0
                - else
                    - F(new_x) < 0
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
            - kernel_function == a kernel function object type as defined at the
              top of dlib/svm/kernel_abstract.h
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
            - kernel_function == a kernel function object type as defined at the
              top of dlib/svm/kernel_abstract.h
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


