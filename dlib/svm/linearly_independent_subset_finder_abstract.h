// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_LISf_ABSTRACT_
#ifdef DLIB_LISf_ABSTRACT_

#include "../algs.h"
#include "../serialize.h"
#include "kernel_abstract.h"

namespace dlib
{

    template <
        typename kernel_type
        >
    class linearly_independent_subset_finder
    {
        /*!
            REQUIREMENTS ON kernel_type
                is a kernel function object as defined in dlib/svm/kernel_abstract.h 

            INITIAL VALUE
                - dictionary_size() == 0

            WHAT THIS OBJECT REPRESENTS
                This is an implementation of an online algorithm for recursively finding a
                set (aka dictionary) of linearly independent vectors in a kernel induced 
                feature space.  To use it you decide how large you would like the dictionary 
                to be and then you feed it sample points.  

                The implementation uses the Approximately Linearly Dependent metric described 
                in the paper The Kernel Recursive Least Squares Algorithm by Yaakov Engel to 
                decide which points are more linearly independent than others.  The metric is 
                simply the squared distance between a test point and the subspace spanned by 
                the set of dictionary vectors.

                Each time you present this object with a new sample point (via this->add()) 
                it calculates the projection distance and if it is sufficiently large then this 
                new point is included into the dictionary.  Note that this object can be configured 
                to have a maximum size.  Once the max dictionary size is reached each new point 
                kicks out a previous point.  This is done by removing the dictionary vector that 
                has the smallest projection distance onto the others.  That is, the "least linearly 
                independent" vector is removed to make room for the new one.
        !*/

    public:
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;

        linearly_independent_subset_finder (
        );
        /*!
            ensures
                - #minimum_tolerance() == 0.001 
                - this object is properly initialized
                - #get_kernel() == kernel_type()  (i.e. whatever the default is for the supplied kernel) 
                - #max_dictionary_size() == 100 
        !*/

        linearly_independent_subset_finder (
            const kernel_type& kernel_, 
            unsigned long max_dictionary_size,
            scalar_type min_tolerance = 0.001
        );
        /*!
            requires
                - min_tolerance > 0
            ensures
                - #minimum_tolerance() == min_tolerance
                - this object is properly initialized
                - #get_kernel() == kernel_
                - #max_dictionary_size() == max_dictionary_size_
        !*/

        const kernel_type& get_kernel (
        ) const;
        /*!
            ensures
                - returns a const reference to the kernel used by this object
        !*/

        unsigned long max_dictionary_size(
        ) const;
        /*!
            ensures
                - returns the maximum number of dictionary vectors this object
                  will accumulate.  That is, dictionary_size() will never be
                  greater than max_dictionary_size().
        !*/

        scalar_type minimum_tolerance(
        ) const;
        /*!
            ensures
                - returns the minimum tolerance to use for the approximately linearly dependent 
                  test used for dictionary vector selection (see KRLS paper for ALD details).  
                  In other words, this is the minimum threshold for how linearly independent 
                  a sample must be for it to be considered for addition to the dictionary.  
                  Moreover, bigger values of this field will make the algorithm run faster but 
                  might give less accurate results.
                - The exact meaning of the tolerance parameter is the following: 
                  Imagine that we have an empirical_kernel_map that contains all the current 
                  dictionary vectors.  Then the tolerance is the minimum projection error 
                  (as given by empirical_kernel_map::project()) required to cause us to 
                  include a new vector in the dictionary.  So each time you call add() this 
                  object basically just computes the projection error for that new sample 
                  and if it is larger than the tolerance then that new sample becomes part 
                  of the dictionary.  
        !*/

        void set_minimum_tolerance (
            scalar_type min_tolerance 
        );
        /*!
            requires
                - min_tolerance > 0
            ensures
                - #minimum_tolerance() == min_tol
        !*/

        void clear_dictionary (
        );
        /*!
            ensures
                - clears out all the data (e.g. #dictionary_size() == 0)
        !*/

        bool add (
            const sample_type& x
        );
        /*!
            ensures
                - if (x is sufficiently linearly independent of the vectors already in this object) then
                    - adds x into the dictionary
                    - (*this)[#dictionary_size()-1] == x
                    - returns true
                    - if (dictionary_size() < max_dictionary_size()) then
                        - #dictionary_size() == dictionary_size() + 1
                    - else
                        - #dictionary_size() == dictionary_size() 
                          (i.e. the number of vectors in this object doesn't change)
                        - the least linearly independent vector in this object is removed
                - else
                    - returns false
        !*/

        scalar_type projection_error (
            const sample_type& x
        ) const;
        /*!
            ensures
                - returns the squared distance between x and the subspace spanned by 
                  the set of dictionary vectors.  (e.g. this is the same number that
                  gets returned by the empirical_kernel_map::project() function's 
                  projection_error argument when the ekm is loaded with the dictionary
                  vectors.)
                - Note that if the dictionary is empty then the return value is
                  equal to get_kernel(x,x).
        !*/

        void swap (
            linearly_independent_subset_finder& item
        );
        /*!
            ensures
                - swaps *this with item
        !*/

        unsigned long dictionary_size (
        ) const;
        /*!
            ensures
                - returns the number of vectors in the dictionary.  
        !*/

        const sample_type& operator[] (
            unsigned long index
        ) const;
        /*!
            requires
                - index < dictionary_size()
            ensures
                - returns the index'th element in the set of linearly independent 
                  vectors contained in this object.
        !*/

        const matrix<sample_type,0,1,mem_manager_type> get_dictionary (
        ) const;
        /*!
            ensures
                - returns a column vector that contains all the dictionary
                  vectors in this object.
        !*/

        const matrix<scalar_type,0,0,mem_manager_type>& get_kernel_matrix (
        ) const;
        /*!
            ensures
                - returns a matrix K such that:
                    - K.nr() == K.nc() == dictionary_size()
                    - K == kernel_matrix(get_kernel(), get_dictionary())
                      i.e. K == the kernel matrix for the dictionary vectors
        !*/

        const matrix<scalar_type,0,0,mem_manager_type>& get_inv_kernel_marix (
        ) const;
        /*!
            ensures
                - if (dictionary_size() != 0)
                    - returns inv(get_kernel_matrix())
                - else
                    - returns an empty matrix
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    void swap(
        linearly_independent_subset_finder<kernel_type>& a, 
        linearly_independent_subset_finder<kernel_type>& b
    ) { a.swap(b); }
    /*!
        provides a global swap function
    !*/

    template <
        typename kernel_type
        >
    void serialize (
        const linearly_independent_subset_finder<kernel_type>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for linearly_independent_subset_finder objects
    !*/

    template <
        typename kernel_type 
        >
    void deserialize (
        linearly_independent_subset_finder<kernel_type>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for linearly_independent_subset_finder objects
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_LISf_ABSTRACT_

