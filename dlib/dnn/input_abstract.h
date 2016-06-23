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

        EXAMPLE_INPUT_LAYER (
            const EXAMPLE_INPUT_LAYER& item
        );
        /*!
            ensures
                - EXAMPLE_INPUT_LAYER objects are copy constructable
        !*/

        EXAMPLE_INPUT_LAYER(
            const some_other_input_layer_type& item
        );
        /*!
            ensures
                - Constructs this object from item.  This form of constructor is optional
                  but it allows you to provide a conversion from one input layer type to
                  another.  For example, the following code is valid only if my_input_layer2 can
                  be constructed from my_input_layer1:
                    relu<fc<relu<fc<my_input_layer1>>>> my_dnn1;
                    relu<fc<relu<fc<my_input_layer2>>>> my_dnn2(my_dnn1);
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

    std::ostream& operator<<(std::ostream& out, const EXAMPLE_INPUT_LAYER& item);
    /*!
        print a string describing this layer.
    !*/

    void to_xml(const EXAMPLE_INPUT_LAYER& item, std::ostream& out);
    /*!
        This function is optional, but required if you want to print your networks with
        net_to_xml().  Therefore, to_xml() prints a layer as XML.
    !*/

    void serialize(const EXAMPLE_INPUT_LAYER& item, std::ostream& out);
    void deserialize(EXAMPLE_INPUT_LAYER& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

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
                - If the input data contains pixels of type unsigned char, rgb_pixel, or
                  other pixel types with a basic_pixel_type of unsigned char then each
                  value written to the output tensor is first divided by 256.0 so that the
                  resulting outputs are all in the range [0,1].
        !*/
    };

// ----------------------------------------------------------------------------------------

    class input_rgb_image
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This input layer works with RGB images of type matrix<rgb_pixel>.  It is
                very similar to the dlib::input layer except that it allows you to subtract
                the average color value from each color channel when converting an image to
                a tensor.
        !*/
    public:
        typedef matrix<rgb_pixel> input_type;
        const static unsigned int sample_expansion_factor = 1;

        input_rgb_image (
        );
        /*!
            ensures
                - #get_avg_red()   == 122.782
                - #get_avg_green() == 117.001
                - #get_avg_blue()  == 104.298
        !*/

        input_rgb_image (
            float avg_red,
            float avg_green,
            float avg_blue
        ); 
        /*!
            ensures
                - #get_avg_red() == avg_red
                - #get_avg_green() == avg_green
                - #get_avg_blue() == avg_blue
        !*/

        float get_avg_red(
        ) const;
        /*!
            ensures
                - returns the value subtracted from the red color channel.
        !*/

        float get_avg_green(
        ) const;
        /*!
            ensures
                - returns the value subtracted from the green color channel.
        !*/

        float get_avg_blue(
        ) const;
        /*!
            ensures
                - returns the value subtracted from the blue color channel.
        !*/

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
                - The input range should contain images that all have the same
                  dimensions.
            ensures
                - Converts the iterator range into a tensor and stores it into #data.  In
                  particular, if the input images have R rows, C columns then we will have:
                    - #data.num_samples() == std::distance(ibegin,iend)
                    - #data.nr() == R
                    - #data.nc() == C
                    - #data.k() == 3
                  Moreover, each color channel is normalized by having its average value
                  subtracted (according to get_avg_red(), get_avg_green(), or
                  get_avg_blue()) and then is divided by 256.0.
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <size_t NR, size_t NC=NR>
    class input_rgb_image_sized 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This layer has an interface and behavior identical to input_rgb_image
                except that it requires input images to have NR rows and NC columns.  This
                is checked by a DLIB_CASSERT inside to_tensor().

                You can also convert between input_rgb_image and input_rgb_image_sized by
                copy construction or assignment.
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_INPUT_ABSTRACT_H_

