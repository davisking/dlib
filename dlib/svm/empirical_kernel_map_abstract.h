// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_EMPIRICAL_KERNEl_MAP_ABSTRACT_H_
#ifdef DLIB_EMPIRICAL_KERNEl_MAP_ABSTRACT_H_

#include <vector>
#include "../matrix.h"
#include "kernel_abstract.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename kern_type
        >
    class empirical_kernel_map
    {
        /*!
            REQUIREMENTS ON kern_type
                - must be a kernel function object as defined in dlib/svm/kernel_abstract.h

            INITIAL VALUE
                - out_vector_size() == 0

            WHAT THIS OBJECT REPRESENTS
                TODO
        !*/
    public:

        typedef kern_type kernel_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;

        struct empirical_kernel_map_error : public error;
        /*!
            This is an exception class used to indicate a failure to create a 
            kernel map from data given by the user.
        !*/

        template <typename EXP>
        void load(
            const kernel_type& kernel,
            const matrix_exp<EXP>& samples
        );
        /*!
            requires
                - samples.size() > 0
            ensures
                - 0 < #out_vector_size() <= samples.size()
                - #get_kernel() == kernel
                - TODO
            throws
                - empirical_kernel_map_error
                    This exception is thrown if we are unable to create a kernel map.
                    If this happens then this object will revert back to its initial value.
        !*/

        void load(
            const kernel_type& kernel,
            const std::vector<sample_type>& samples
        );
        /*!
            requires
                - samples.size() > 0
            ensures
                - performs load(kernel,vector_to_matrix(samples)).  I.e. This function
                  does the exact same thing as the above load() function but lets you use
                  a std::vector of samples in addition to a row/column matrix of samples.
        !*/

        const kernel_type get_kernel (
        ) const;
        /*!
            requires
                - out_vector_size() != 0
            ensures
                - returns a copy of the kernel used by this object
        !*/

        long out_vector_size (
        ) const;
        /*!
            ensures
                - if (this object has been loaded with sample basis functions) then
                    - returns the dimensionality of the space the kernel map projects
                      new data samples into via the project() function.
                - else
                    - returns 0
        !*/

        const matrix<scalar_type,0,1,mem_manager_type>& project (
            const sample_type& sample 
        ) const;
        /*!
            requires
                - out_vector_size() != 0
            ensures
                - takes the given sample and maps it into the kernel feature space
                  of out_vector_size() dimensions defined by this kernel map and 
                  returns the resulting vector.
        !*/

        void swap (
            empirical_kernel_map& item
        );
        /*!
            ensures
                - swaps the state of *this and item
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    void swap (
        empirical_kernel_map<kernel_type>& a,
        empirical_kernel_map<kernel_type>& b
    ) { a.swap(b); }
    /*!
        provides a global swap function
    !*/

    template <
        typename kernel_type
        >
    void serialize (
        const empirical_kernel_map<kernel_type>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for empirical_kernel_map objects
    !*/

    template <
        typename kernel_type
        >
    void deserialize (
        empirical_kernel_map<kernel_type>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for empirical_kernel_map objects
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_EMPIRICAL_KERNEl_MAP_ABSTRACT_H_

