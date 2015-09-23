// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_INPUT_H_
#define DLIB_DNn_INPUT_H_

#include "../matrix.h"
#include "../pixel.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename T>
    class input 
    {
    public:

        // sample_expansion_factor must be > 0
        const static unsigned int sample_expansion_factor = 1;
        typedef T input_type;

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const
        /*!
            requires
                - [begin, end) is an iterator range over input_type objects.
            ensures
                - Converts the iterator range into a tensor and stores it into #data.
                - Normally you would have #data.num_samples() == distance(begin,end) but
                  you can also expand the output by some integer factor so long as the loss
                  you use can deal with it correctly.
                - #data.num_samples() == distance(begin,end)*sample_expansion_factor. 
        !*/
        {
            // initialize data to the right size to contain the stuff in the iterator range.

            for (input_iterator i = begin; i != end; ++i)
            {
                matrix<rgb_pixel> temp = *i;
                // now copy *i into the right part of data.
            }
        }
    };

// ----------------------------------------------------------------------------------------

    template <typename T,long NR, typename MM, typename L>
    class input<matrix<T,NR,1,MM,L>> 
    {
    public:

        // TODO, maybe we should only allow T to be float?  Seems kinda pointless to allow
        // double. Don't forget to remove the matrix_cast if we enforce just float.
        typedef matrix<T,NR,1,MM,L> input_type;
        const static unsigned int sample_expansion_factor = 1;

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const
        /*!
            requires
                - [begin, end) is an iterator range over input_type objects.
            ensures
                - converts the iterator range into a tensor and stores it into #data.
                - Normally you would have #data.num_samples() == distance(begin,end) but
                  you can also expand the output by some integer factor so long as the loss
                  you use can deal with it correctly.
                - #data.num_samples() == distance(begin,end)*sample_expansion_factor. 
        !*/
        {
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(begin,end), 1, 1, begin->size());

            unsigned long idx = 0;
            for (input_iterator i = begin; i != end; ++i)
            {
                data.set_sample(idx++, matrix_cast<float>(*i));
            }
        }
    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    class input2
    {
    public:

        input2(){}

        input2(const input<T>&) {}

        typedef T input_type;
        const static unsigned int sample_expansion_factor = 1;

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const
        /*!
            requires
                - [begin, end) is an iterator range over T objects.
            ensures
                - converts the iterator range into a tensor and stores it into #data.
                - Normally you would have #data.num_samples() == distance(begin,end) but
                  you can also expand the output by some integer factor so long as the loss
                  you use can deal with it correctly.
                - #data.num_samples() == distance(begin,end)*K where K is an integer >= 1. 
        !*/
        {
            // initialize data to the right size to contain the stuff in the iterator range.

            for (input_iterator i = begin; i != end; ++i)
            {
                matrix<rgb_pixel> temp = *i;
                // now copy *i into the right part of data.
            }
        }
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_INPUT_H_

