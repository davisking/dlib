

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>


using namespace std;
using namespace dlib;


/*
    Training differences with dnn_mmod_ex.cpp

    A slightly bigger network architecture.  Also, to train you must replace the affine layers with bn_con layers.

    mmod_options options(training_labels, 80*80);
    instead of 
    mmod_options options(face_boxes_train, 40*40);

    trainer.set_iterations_without_progress_threshold(8000);
    instead of 
    trainer.set_iterations_without_progress_threshold(300);

    random cropper was left at its default settings,  So we didn't call these functions:
    cropper.set_chip_dims(200, 200);
    cropper.set_min_object_height(0.2);
*/


// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------


int main(int argc, char** argv) try
{
    if (argc < 3)
    {
        cout << "Give the path to the examples/faces directory as the argument to this" << endl;
        cout << "program.  For example, if you are in the examples folder then execute " << endl;
        cout << "this program by running: " << endl;
        cout << "   ./fhog_object_detector_ex faces" << endl;
        cout << endl;
        return 0;
    }


    net_type net;
    deserialize(argv[1]) >> net;  

    image_window win;
    for (int i = 2; i < argc; ++i)
    {
        matrix<rgb_pixel> img;
        load_image(img, argv[i]);

        // Upsampling the image will allow us to detect smaller faces but will cause the
        // program to use more RAM and run longer.
        pyramid_up(img); 
        pyramid_up(img);

        // Note that you can process a bunch of images in a std::vector at once and it runs
        // faster, since this will form mini-batches of images and therefore get better
        // parallelism out of your GPU hardware.  However, all the images must be the same
        // size.  To avoid this requirement on images being the same size we process them
        // individually in this example.
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        cin.get();
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




