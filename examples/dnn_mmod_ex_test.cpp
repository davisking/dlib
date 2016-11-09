#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <time.h>

using namespace std;
using namespace dlib;

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;
template <unsigned long N, typename SUBNET> using downsampler  = relu<bn_con<con5d<N, relu<bn_con<con5d<N, relu<bn_con<con5d<N,SUBNET>>>>>>>>>;
template <unsigned long N, typename SUBNET> using rcon3  = relu<bn_con<con3<N,SUBNET>>>;
using net_type  = loss_mmod<con<1,6,6,1,1,rcon3<32,rcon3<32,rcon3<32,downsampler<32,input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

int main(int argc, char** argv) try
{

    if (argc != 2)
    {
        cout << "Give the path to the examples/faces directory as the argument to this" << endl;
        cout << "program.  For example, if you are in the examples folder then execute " << endl;
        cout << "this program by running: " << endl;
        cout << "   ./dnn_mmod_ex_test faces" << endl;
        cout << endl;
        return 0;
    }
    const std::string faces_directory = argv[1];
    std::vector<matrix<rgb_pixel>> images_train, images_test;
    std::vector<std::vector<mmod_rect>> face_boxes_train, face_boxes_test;
    load_image_dataset(images_test, face_boxes_test, faces_directory+"/testing.xml");
    cout << "num testing images:  " << images_test.size() << endl;
    net_type net;
    deserialize("mmod_network.dat") >> net;
    // Now lets run the detector on the testing images and look at the outputs.
    cout << "starting detection";
    image_window win;
    for (auto&& img : images_test)
    {
        //pyramid_down(img);
        clock_t tStart = clock();
        auto dets = net(img);
        cout << dets.size() << '\n'; // prints 10
        printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        cin.get();
    }
    cout << "ending detection";
    return 0;

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}