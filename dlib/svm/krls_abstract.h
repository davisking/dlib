// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_KRLs_ABSTRACT_
#ifdef DLIB_KRLs_ABSTRACT_

#include <cmath>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "../serialize.h"

namespace dlib
{

    template <
        typename kernel_type
        >
    class krls
    {
        /*!
            INITIAL VALUE
                - dictionary_size() == 0

            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the kernel recursive least squares algorithm 
                described in the paper:
                    The Kernel Recursive Least Squares Algorithm by Yaakov Engel.

                The long and short of this algorithm is that it is an online kernel based 
                regression algorithm.  You give it samples (x,y) and it learns the function
                f(x) == y.  For a detailed description of the algorithm read the above paper.
        !*/

    public:
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;


        explicit krls (
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
            ensures
                - #get_tolerance() == tolerance_
        !*/

        scalar_type get_tolerance(
        ) const;
        /*!
            ensures
                - returns the tolerance to use for the approximately linearly dependent 
                  test in the KRLS algorithm.  This is a number which governs how 
                  accurately this object will approximate the decision function it is 
                  learning.  Smaller values generally result in a more accurate 
                  estimate while also resulting in a bigger set of support vectors in 
                  the learned decision function.  Bigger tolerances values result in a 
                  less accurate decision function but also in less support vectors.
        !*/

        void clear (
        );
        /*!
            ensures
                - clears out all learned data and puts this object back to its
                  initial state.
                  (e.g. #get_decision_function().support_vectors.size() == 0)
                - #get_tolerance() == get_tolerance()
                  (i.e. doesn't change the value of the tolerance)
        !*/

        scalar_type operator() (
            const sample_type& x
        ) const;
        /*!
            ensures
                - returns the current y estimate for the given x
        !*/

        void train (
            const sample_type& x,
            scalar_type y
        );
        /*!
            ensures
                - trains this object that the given x should be mapped to the given y
        !*/

        void swap (
            krls& item
        );
        /*!
            ensures
                - swaps *this with item
        !*/

        unsigned long dictionary_size (
        ) const;
        /*!
            ensures
                - returns the number of "support vectors" in the dictionary.  That is,
                  returns a number equal to get_decision_function().support_vectors.size()
        !*/

        decision_function<kernel_type> get_decision_function (
        ) const;
        /*!
            ensures
                - returns a decision function F that represents the function learned
                  by this object so far.  I.e. it is the case that:
                    - for all x: F(x) == (*this)(x)
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    void swap(
        krls<kernel_type>& a, 
        krls<kernel_type>& b
    )
    { a.swap(b); }
    /*!
        provides a global swap function
    !*/

    template <
        typename kernel_type
        >
    void serialize (
        const krls<kernel_type>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for krls objects
    !*/

    template <
        typename kernel_type 
        >
    void deserialize (
        krls<kernel_type>& item,
        std::istream& in 
    );
    /*!
        provides serialization support for krls objects
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KRLs_ABSTRACT_

