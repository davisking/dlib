// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to export dlib's pretrained ResNet34 ImageNet
    classifier to ONNX.

    You will need a copy of resnet34_1000_imagenet_classifier.dnn from:
        http://dlib.net/files/resnet34_1000_imagenet_classifier.dnn.bz2

    The exported ONNX graph input is the NCHW tensor accepted by net.forward().
    For dlib image input layers, that means the image preprocessing normally
    performed by the dlib input layer happens outside this ONNX graph.
*/

#include <dlib/dnn.h>

#include <iostream>
#include <string>
#include <vector>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using level1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using level2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,SUBNET>>>>>>;
template <typename SUBNET> using level3 = ares<128,ares<128,ares<128,ares_down<128,SUBNET>>>>;
template <typename SUBNET> using level4 = ares<64,ares<64,ares<64,SUBNET>>>;

using anet_type = loss_multiclass_log<fc<1000,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<affine<con<64,7,7,2,2,
                            input_rgb_image_sized<227>
                            >>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc < 2 || argc > 3)
    {
        cout << "Usage: " << argv[0] << " <resnet34_1000_imagenet_classifier.dnn> [output.onnx]\n";
        return 1;
    }

    const string dnn_filename = argv[1];
    const string onnx_filename = argc == 3 ? argv[2] : "resnet34_1000_imagenet_classifier.onnx";

    anet_type net;
    std::vector<std::string> labels;
    deserialize(dnn_filename) >> net >> labels;

    softmax<anet_type::subnet_type> snet;
    snet.subnet() = net.subnet();

    onnx_export_options options;
    options.input_name = "input";
    options.output_name = "probabilities";
    options.graph_name = "dlib_resnet34_imagenet";
    // To export an ONNX graph that accepts RGB pixel values in float NCHW
    // format and performs dlib's input layer preprocessing internally, use:
    // options.input_mode = onnx_export_input_mode::dlib_input_layer;

    net_to_onnx(snet, onnx_filename, options);

    cout << "Wrote " << onnx_filename << "\n";
    cout << "Input tensor shape: [1,3,227,227]\n";
    cout << "Input tensor semantics: dlib preprocessed NCHW tensor accepted by net.forward().\n";

    /*
        Minimal ONNX Runtime call shape for the default export:

            matrix<rgb_pixel> img, resized;
            load_image(img, image_filename);
            resized.set_size(227, 227);
            resize_image(img, resized);

            std::vector<matrix<rgb_pixel> > images(1, resized);
            resizable_tensor input_tensor;
            snet.to_tensor(images.begin(), images.end(), input_tensor);

            std::array<int64_t,4> shape = {{1, 3, 227, 227}};
            Ort::Value input = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor.host(),
                input_tensor.size(),
                shape.data(),
                shape.size()
            );

            session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);

        If options.input_mode is set to dlib_input_layer above, feed raw RGB
        values as float NCHW data in the 0-255 range instead of input_tensor.
    */
}
catch (std::exception& e)
{
    cout << e.what() << endl;
    return 1;
}

// ----------------------------------------------------------------------------------------
