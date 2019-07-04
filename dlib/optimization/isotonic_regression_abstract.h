// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_ISOTONIC_ReGRESSION_ABSTRACT_H_
#ifdef DLIB_ISOTONIC_ReGRESSION_ABSTRACT_H_

#include <vector>
#include <utility>

namespace dlib
{
    class isotonic_regression
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for performing 1-D isotonic regression. That is, it
                finds the least squares fit of a non-parametric curve to some user supplied
                data, subject to the constraint that the fitted curve is non-decreasing.

                This is done using the fast O(n) pool adjacent violators algorithm.
        !*/

    public:

        template <
            typename const_iterator, 
            typename iterator
            >
        void operator() (
            const_iterator begin,
            const_iterator end,
            iterator obegin
        );
        /*!
            requires
                - [begin,end) is an iterator range of float or doubles or a range of
                  std::pair<T,double> or std::pair<T,float> where T an be anything.
                - obegin points to an iterator range at least std::distance(begin,end) elements. 
                - obegin points to an iterator range of objects of type float, double, std::pair<T,float>, or std::pair<T,double>.
            ensures
                - Given the range of real values stored in [begin,end), this method performs isotonic regression
                  on this data and writes the results to obegin.  To be specific:
                    - let IN refer to the input values stored in the iterator range [begin,end).
                    - let OUT refer to the output values stored in the iterator range [obegin, obegin+std::distance(begin,end)).
                    - This function populates OUT with values such that the sum_i of
                      (IN[i]-OUT[i])^2 is minimized, subject to the constraint that 
                      OUT[i] <= OUT[i+1], i.e. that OUT is monotonic.
                - It is OK for [begin,end) to overlap with the range pointed to by obegin.
                  That is, this function can run in-place.
                - Note that when the inputs or outputs are std::pairs this algorithm only
                  looks at the .second field of the pair.  It therefore still treats these
                  iterator ranges as ranges of reals since it only looks at the .second
                  field, which is a real number.  The .first field is entirely ignored.
        !*/

        void operator() (
            std::vector<double>& vect
        ) { (*this)(vect.begin(), vect.end(), vect.begin()); }
        /*!
            ensures
                - performs in-place isotonic regression.  Therefore, #vect will contain the
                  isotonic regression of vect.
                - #vect.size() == vect.size()
        !*/

        template <typename T, typename U>
        void operator() (
            std::vector<std::pair<T,U>>& vect
        ) { (*this)(vect.begin(), vect.end(), vect.begin()); }
        /*!
            ensures
                - performs in-place isotonic regression.  Therefore, #vect will contain the
                  isotonic regression of vect.
                - #vect.size() == vect.size()
        !*/


        template <
            typename const_iterator, 
            typename iterator
            >
        void fit_with_linear_output_interpolation (
            const_iterator begin,
            const_iterator end,
            iterator obegin
        );
        /*!
            requires
                - [begin,end) is an iterator range of float or doubles or a range of
                  std::pair<T,double> or std::pair<T,float> where T an be anything.
                - obegin points to an iterator range at least std::distance(begin,end). 
                - obegin points to an iterator range of objects of type float, double, std::pair<T,float>, or std::pair<T,double>.
            ensures
                - This function behaves just like (*this)(begin,end,obegin) except that the
                  output is interpolated.  To explain, note that the optimal output of
                  isotonic regression is a step function.  However, in many applications
                  that isn't really what you want.  You want something smoother.  So
                  fit_with_linear_output_interpolation() does isotonic regression and then
                  linearly interpolates the step function into a piecewise linear function.
        !*/

        void fit_with_linear_output_interpolation (
            std::vector<double>& vect
        ) { fit_with_linear_output_interpolation(vect.begin(), vect.end(), vect.begin()); }
        /*!
            ensures
                - performs in-place isotonic regression.  Therefore, #vect will contain the
                  isotonic regression of vect.
                - #vect.size() == vect.size()
        !*/

        template <typename T, typename U>
        void fit_with_linear_output_interpolation (
            std::vector<std::pair<T,U>>& vect
        ) { fit_with_linear_output_interpolation(vect.begin(), vect.end(), vect.begin()); }
        /*!
            ensures
                - performs in-place isotonic regression.  Therefore, #vect will contain the
                  isotonic regression of vect.
                - #vect.size() == vect.size()
        !*/

    };
}

#endif // DLIB_ISOTONIC_ReGRESSION_ABSTRACT_H_



