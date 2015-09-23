// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_LOSS_H_
#define DLIB_DNn_LOSS_H_

#include "core.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class loss_binary_hinge_ 
    {
    public:

        const static unsigned int sample_expansion_factor = 1;
        typedef double label_type;

        // Implementing to_label() is optional.  If you don't do it then it just means the
        // automatic operator() mapping from tensors to outputs is missing from the net object.
        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const SUB_TYPE& sub,
            label_iterator iter
        ) const
        /*!
            requires
                - SUB_NET implements the SUB_NET interface defined at the top of layers_abstract.h.
                - sub.get_output().num_samples() must be a multiple of sample_expansion_factor.
                - iter == an iterator pointing to the beginning of a range of
                  sub.get_output().num_samples()/sample_expansion_factor elements.  In
                  particular, they must be label_type elements.
        !*/
        {
            const tensor& output_tensor = sub.get_output();
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1 && 
                         output_tensor.k() == 1,"");
            DLIB_CASSERT(output_tensor.num_samples()%sample_expansion_factor == 0,"");

            const float* out_data = output_tensor.host();
            for (unsigned long i = 0; i < output_tensor.num_samples(); ++i)
            {
                *iter++ = out_data[i];
            }
        }

        template <
            typename label_iterator,
            typename SUB_NET
            >
        double compute_loss (
            const tensor& input_tensor,
            label_iterator truth, // TODO, this parameter is optional.
            SUB_NET& sub
        ) const
        /*!
            requires
                - SUB_NET implements the SUB_NET interface defined at the top of layers_abstract.h.
                - input_tensor was given as input to the network sub and the outputs are now
                  visible in sub.get_output(), sub.sub_net().get_output(), etc.
                - input_tensor.num_samples() must be a multiple of sample_expansion_factor.
                - input_tensor.num_samples() == sub.get_output().num_samples() == grad.num_samples()
                - truth == an iterator pointing to the beginning of a range of
                  input_tensor.num_samples()/sample_expansion_factor elements.  In particular,
                  they must be label_type elements.
                - sub.get_gradient_input() has the same dimensions as sub.get_output().
                - for all valid i:
                    - *(truth+i/sample_expansion_factor) is the label of the ith sample in
                      sub.get_output().
            ensures
                - #sub.get_gradient_input() == the gradient of the loss with respect to
                  sub.get_output().
        !*/
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();

            // TODO, throw an exception instead of asserting, probably...
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples(),"");
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples(),"");
            DLIB_CASSERT(output_tensor.nr() == 1 && 
                         output_tensor.nc() == 1 && 
                         output_tensor.k() == 1,"");

            // The loss we output is the average loss over the mini-batch.
            const double scale = 1.0/output_tensor.num_samples();
            double loss = 0;
            const float* out_data = output_tensor.host();
            float* g = grad.host();
            for (unsigned long i = 0; i < output_tensor.num_samples(); ++i)
            {
                const float y = *truth++;
                const float temp = 1-y*out_data[i];
                if (temp > 0)
                {
                    loss += scale*temp;
                    g[i] = -scale*y;
                }
                else
                {
                    g[i] = 0;
                }
            }
            return loss;
        }

    };

// ----------------------------------------------------------------------------------------

    template <typename SUB_NET>
    using loss_binary_hinge = add_loss_layer<loss_binary_hinge_, SUB_NET>;

// ----------------------------------------------------------------------------------------

    class loss_no_label_ 
    {
    public:

        //typedef int label_type;

        const static unsigned int sample_expansion_factor = 1;


        template <
            typename SUB_NET
            >
        double compute_loss (
            const tensor& input_tensor,
            SUB_NET& sub
        ) const
        /*!
            requires
                - SUB_NET implements the SUB_NET interface defined at the top of layers_abstract.h.
                - input_tensor was given as input to the network sub and the outputs are now
                  visible in sub.get_output(), sub.sub_net().get_output(), etc.
                - input_tensor.num_samples() must be a multiple of sample_expansion_factor.
                - input_tensor.num_samples() == sub.get_output().num_samples() == grad.num_samples()
                - truth == an iterator pointing to the beginning of a range of
                  input_tensor.num_samples()/sample_expansion_factor elements.  In particular,
                  they must be label_type elements.
                - sub.get_gradient_input() has the same dimensions as sub.get_output().
                - for all valid i:
                    - *(truth+i/sample_expansion_factor) is the label of the ith sample in
                      sub.get_output().
            ensures
                - #sub.get_gradient_input() == the gradient of the loss with respect to
                  sub.get_output().
        !*/
        {
            return 0;
        }

    };

// ----------------------------------------------------------------------------------------

    template <typename SUB_NET>
    using loss_no_label = add_loss_layer<loss_no_label_, SUB_NET>;

// ----------------------------------------------------------------------------------------

}

#endif // #define DLIB_DNn_LOSS_H_


