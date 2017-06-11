// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to classify an image into one of the 1000 imagenet
    categories using the deep learning tools from the dlib C++ Library.  We will
    use the pretrained ResNet34 model available on the dlib website.

    The ResNet34 architecture is from the paper Deep Residual Learning for Image
    Recognition by He, Zhang, Ren, and Sun.  The model file that comes with dlib
    was trained using the dnn_imagenet_train_ex.cpp program on a Titan X for
    about 2 weeks.  This pretrained model has a top5 error of 7.572% on the 2012
    imagenet validation dataset.

    For an introduction to dlib's DNN module read the dnn_introduction_ex.cpp and
    dnn_introduction2_ex.cpp example programs.

    
    Finally, these tools will use CUDA and cuDNN to drastically accelerate
    network training and testing.  CMake should automatically find them if they
    are installed and configure things appropriately.  If not, the program will
    still run but will be much slower to execute.
*/



#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>

using namespace std;
using namespace dlib;
 
// ----------------------------------------------------------------------------------------

// This block of statements defines the resnet-34 network

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

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

rectangle make_random_cropping_rect_resnet(
    const matrix<rgb_pixel>& img,
    dlib::rand& rnd
)
{
    // figure out what rectangle we want to crop from the image
    double mins = 0.466666666, maxs = 0.875;
    auto scale = mins + rnd.get_random_double()*(maxs-mins);
    auto size = scale*std::min(img.nr(), img.nc());
    rectangle rect(size, size);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                 rnd.get_random_32bit_number()%(img.nr()-rect.height()));
    return move_rect(rect, offset);
}

// ----------------------------------------------------------------------------------------

void randomly_crop_images (
    const matrix<rgb_pixel>& img,
    dlib::array<matrix<rgb_pixel>>& crops,
    dlib::rand& rnd,
    long num_crops
)
{
    std::vector<chip_details> dets;
    for (long i = 0; i < num_crops; ++i)
    {
        auto rect = make_random_cropping_rect_resnet(img, rnd);
        dets.push_back(chip_details(rect, chip_dims(227,227)));
    }

    extract_image_chips(img, dets, crops);

    for (auto&& img : crops)
    {
        // Also randomly flip the image
        if (rnd.get_random_double() > 0.5)
            img = fliplr(img);

        // And then randomly adjust the colors.
        apply_random_color_offset(img, rnd);
    }
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc == 1)
    {
        cout << "Give this program image files as command line arguments.\n" << endl;
        cout << "You will also need a copy of the file resnet34_1000_imagenet_classifier.dnn " << endl;
        cout << "available at http://dlib.net/files/resnet34_1000_imagenet_classifier.dnn.bz2" << endl;
        cout << endl;
        return 1;
    }

    std::vector<string> labels;
    anet_type net;
    deserialize("resnet34_1000_imagenet_classifier.dnn") >> net >> labels;

    // Make a network with softmax as the final layer.  We don't have to do this
    // if we just want to output the single best prediction, since the anet_type
    // already does this.  But if we instead want to get the probability of each
    // class as output we need to replace the last layer of the network with a
    // softmax layer, which we do as follows:
    softmax<anet_type::subnet_type> snet; 
    snet.subnet() = net.subnet();

    dlib::array<matrix<rgb_pixel>> images;
    matrix<rgb_pixel> img, crop;

    dlib::rand rnd;
    image_window win;

    // Read images from the command prompt and print the top 5 best labels for each.
    for (int i = 1; i < argc; ++i)
    {
        load_image(img, argv[i]);
        const int num_crops = 16;
        // Grab 16 random crops from the image.  We will run all of them through the
        // network and average the results.
        randomly_crop_images(img, images, rnd, num_crops);
        // p(i) == the probability the image contains object of class i.
        matrix<float,1,1000> p = sum_rows(mat(snet(images.begin(), images.end())))/num_crops;

        win.set_image(img);
        // Print the 5 most probable labels
        for (int k = 0; k < 5; ++k)
        {
            unsigned long predicted_label = index_of_max(p);
            cout << p(predicted_label) << ": " << labels[predicted_label] << endl;
            p(predicted_label) = 0;
        }

        cout << "Hit enter to process the next image";
        cin.get();
    }

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

