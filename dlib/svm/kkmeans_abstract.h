// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_KKMEANs_ABSTRACT_
#ifdef DLIB_KKMEANs_ABSTRACT_

#include <cmath>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "../serialize.h"
#include "kernel_abstract.h"
#include "kcentroid_abstract.h"
#include "../noncopyable.h"

namespace dlib
{

    template <
        typename kernel_type : public noncopyable
        >
    class kkmeans
    {
        /*!
            REQUIREMENTS ON kernel_type
                is a kernel function object as defined in dlib/svm/kernel_abstract.h 

            INITIAL VALUE
                - number_of_centers() == 1

            WHAT THIS OBJECT REPRESENTS
                This is an implementation of a kernelized k-means clustering algorithm.  
                It performs k-means clustering by using the kcentroid object.  
        !*/

    public:
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;

        kkmeans (
            const kcentroid<kernel_type>& kc_ 
        );
        /*!
            ensures
                - #number_of_centers() == 1
                - #get_kcentroid(0) == a copy of kc_
        !*/

        ~kkmeans(
        );
        /*!
            ensures
                - all resources associated with *this have been released
        !*/

        void set_kcentroid (
            const kcentroid<kernel_type>& kc_
        );
        /*!
            ensures
                - for all idx:  
                    - #get_kcentroid(idx) == a copy of kc_
        !*/

        const kcentroid<kernel_type>& get_kcentroid (
            unsigned long i
        ) const;
        /*!
            ensures
                - returns a const reference to the ith kcentroid object contained in
                  this object.  Each kcentroid represents one of the centers found
                  by the k-means clustering algorithm.
        !*/

        void set_number_of_centers (
            unsigned long num
        );
        /*!
            requires
                - num > 0
            ensures
                - #number_of_centers() == num
        !*/

        unsigned long number_of_centers (
        ) const;
        /*!
            ensures
                - returns the number of centers used in this instance of the k-means clustering
                  algorithm.
        !*/

        template <
            typename matrix_type
            >
        void train (
            const matrix_type& samples,
            const matrix_type& initial_centers 
        );
        /*!
            requires
                - matrix_type::type == sample_type  (i.e. matrix_type should contain sample_type objects)
                - initial_centers.nc() == 1         (i.e. must be a column vector)
                - samples.nc() == 1                 (i.e. must be a column vector)
                - initial_centers.nr() == number_of_centers()
            ensures
                - performs k-means clustering of the given set of samples.  The initial center points
                  are taken from the initial_centers argument.
                - After this function finishes you can call the operator() function below
                  to determine which centroid a given sample is closest to.
        !*/

        unsigned long operator() (
            const sample_type& sample
        ) const;
        /*!
            ensures
                - returns a number idx such that:
                    - idx < number_of_centers()
                    - get_kcentroid(idx) == the centroid that is closest to the given
                      sample.
        !*/

        void swap (
            kkmeans& item
        );
        /*!
            ensures
                - swaps *this and item
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    void swap(
        kkmeans<kernel_type>& a, 
        kkmeans<kernel_type>& b
    ) { a.swap(b); }
    /*!
        provides a global swap function
    !*/

    template <
        typename kernel_type
        >
    void serialize (
        const kkmeans<kernel_type>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for kkmeans objects
    !*/

    template <
        typename kernel_type 
        >
    void deserialize (
        kkmeans<kernel_type>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for kkmeans objects
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KKMEANs_ABSTRACT_

