// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_INPUT_ABSTRACT_H_
#ifdef DLIB_DNn_INPUT_ABSTRACT_H_

#include "../matrix.h"
#include "../pixel.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    class EXAMPLE_INPUT_LAYER
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Each deep neural network model in dlib begins with an input layer. The job
                of the input layer is to convert an input_type into a tensor.  Nothing more
                and nothing less.  
                
                Note that there is no dlib::EXAMPLE_INPUT_LAYER type.  It is shown here
                purely to document the interface that an input layer object must implement.
                If you are using some kind of image or matrix object as your input_type
                then you can use the provided dlib::input layer defined below.  Otherwise,
                you need to define your own custom input layer.
        !*/
    public:

        EXAMPLE_INPUT_LAYER(
        );
        /*!
            ensures
                - Default constructs this object.  This function is not required to do
                  anything in particular but it must exist, that is, it is required that
                  layer objects be default constructable. 
        !*/

        EXAMPLE_INPUT_LAYER(
            const some_other_input_layer_type& item
        );
        /*!
            ensures
                - Constructs this object from item.  This form of constructor is optional
                  but it allows you to provide a conversion from one input layer type to
                  another.  For example, the following code is valid only if my_input2 can
                  be constructed from my_input1:
                    relu<fc<relu<fc<my_input1>>>> my_dnn1;
                    relu<fc<relu<fc<my_input2>>>> my_dnn2(my_dnn1);
                  This kind of pattern is useful if you want to use one type of input layer
                  during training but a different type of layer during testing since it
                  allows you to easily convert between related deep neural network types.  
        !*/

        // sample_expansion_factor must be > 0
        const static unsigned int sample_expansion_factor;
        typedef whatever_type_to_tensor_expects input_type;

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const;
        /*!
            requires
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
            ensures
                - Converts the iterator range into a tensor and stores it into #data.
                - #data.num_samples() == distance(ibegin,iend)*sample_expansion_factor. 
                  Normally you would have #data.num_samples() == distance(ibegin,iend) but
                  you can also expand the output by some integer factor so long as the loss
                  you use can deal with it correctly.
                - The data in the ith sample of #data corresponds to the input_type object
                  *(ibegin+i/sample_expansion_factor).
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    class input 
    {
        /*!
            REQUIREMENTS ON T
                T is a matrix or array2d object and it must contain some kind of pixel
                type.  I.e. pixel_traits<T::type> must be defined. 

            WHAT THIS OBJECT REPRESENTS
                This is a basic input layer that simply copies images into a tensor.  
        !*/

    public:
        const static unsigned int sample_expansion_factor = 1;
        typedef T input_type;

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const;
        /*!
            requires
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
                - The input range should contain image objects that all have the same
                  dimensions.
            ensures
                - Converts the iterator range into a tensor and stores it into #data.  In
                  particular, if the input images have R rows, C columns, and K channels
                  (where K is given by pixel_traits::num) then we will have:
                    - #data.num_samples() == std::distance(ibegin,iend)
                    - #data.nr() == R
                    - #data.nc() == C
                    - #data.k() == K
                  For example, a matrix<float,3,3> would turn into a tensor with 3 rows, 3
                  columns, and k()==1.  Or a matrix<rgb_pixel,4,5> would turn into a tensor
                  with 4 rows, 5 columns, and k()==3 (since rgb_pixels have 3 channels).
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_INPUT_ABSTRACT_H_

