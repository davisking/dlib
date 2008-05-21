// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_ONE_CLASs_ABSTRACT_
#ifdef DLIB_ONE_CLASs_ABSTRACT_

#include <cmath>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "../serialize.h"
#include "kernel_abstract.h"

namespace dlib
{

    template <
        typename kernel_type
        >
    class one_class
    {
        /*!
            REQUIREMENTS ON kernel_type
                is a kernel function object as defined in dlib/svm/kernel_abstract.h 

            INITIAL VALUE
                - dictionary_size() == 0
                - max_discount() == 1e6

            WHAT THIS OBJECT REPRESENTS
                This is an implementation of an online algorithm for recursively estimating the
                center of mass of a sequence of training points.  It uses the sparsification technique
                described in the paper The Kernel Recursive Least Squares Algorithm by Yaakov Engel.

                This object then allows you to compute the distance between the center of mass
                and any test points.  So you can use this object to predict how similar a test
                point is to the data this object has been trained on (larger distances from the
                centroid indicate dissimilarity/anomalous points).
        !*/

    public:
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;


        explicit one_class (
            const kernel_type& kernel_, 
            scalar_type tolerance_ = 0.001
        );
        /*!
            ensures
                - this object is properly initialized
                - #get_tolerance() == tolerance_
                - #get_decision_function().kernel_function == kernel_
                  (i.e. this object will use the given kernel function)
        !*/

        void set_tolerance (
            scalar_type tolerance_
        );
        /*!
            requires
                - tolerance_ >= 0
            ensures
                - #get_tolerance() == tolerance_
        !*/

        scalar_type get_tolerance(
        ) const;
        /*!
            ensures
                - returns the tolerance to use for the approximately linearly dependent 
                  test used for sparsification (see the KRLS paper for details).  This is 
                  a number which governs how accurately this object will approximate the 
                  centroid it is learning.  Smaller values generally result in a more accurate 
                  estimate while also resulting in a bigger set of support vectors in 
                  the learned dictionary.  Bigger tolerances values result in a 
                  less accurate estimate but also in less support vectors.
        !*/

        void set_max_discount (
            scalar_type value 
        );
        /*!
            requires
                - value > 0
            ensures
                - #get_max_discount() == value 
        !*/

        scalar_type get_max_discount(
        ) const;
        /*!
            ensures
                - If you have shown this object N samples so far then it has found 
                  the centroid of those N samples.  That is, it has found the average 
                  of all of them in some high dimensional feature space. 
                - if (N <= get_max_discount()) then
                    - The next sample you show this object will be added to the centroid 
                      with a weight of 1/(N+1).  
                - else
                    - The next sample you show this object will be added to the centroid 
                      with a weight of 1/(get_max_discount()+1).  

                - If you think your samples are from a stationary source then you
                  should set the max discount to some really big number.  However, 
                  if you think the source isn't stationary then use a smaller number.
                  This will cause the centroid in this object to be closer to the 
                  centroid of the more recent points.
        !*/

        void clear_dictionary (
        );
        /*!
            ensures
                - clears out all learned data (e.g. #dictionary_size() == 0)
        !*/

        scalar_type operator() (
            const sample_type& x
        ) const;
        /*!
            ensures
                - returns the distance in feature space between the sample x and the
                  current estimate of the centroid of the training samples given
                  to this object so far.
        !*/

        void train (
            const sample_type& x
        );
        /*!
            ensures
                - adds the sample x into the current estimate of the centroid
        !*/

        void swap (
            one_class& item
        );
        /*!
            ensures
                - swaps *this with item
        !*/

        unsigned long dictionary_size (
        ) const;
        /*!
            ensures
                - returns the number of "support vectors" in the dictionary.  
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    void swap(
        one_class<kernel_type>& a, 
        one_class<kernel_type>& b
    ) { a.swap(b); }
    /*!
        provides a global swap function
    !*/

    template <
        typename kernel_type
        >
    void serialize (
        const one_class<kernel_type>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for one_class objects
    !*/

    template <
        typename kernel_type 
        >
    void deserialize (
        one_class<kernel_type>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for one_class objects
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_ONE_CLASs_ABSTRACT_

