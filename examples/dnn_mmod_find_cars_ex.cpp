

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/dir_nav.h>
#include <dlib/time_this.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace dlib;



// the dnn rear view vehicle detector network
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<55,SUBNET>>>;
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main() try
{
    net_type net;
    shape_predictor sp;
    // You can get this file from http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2
    // This network was produced by the dnn_mmod_train_find_cars_ex.cpp example program.
    // As you can see, it also includes a shape_predictor.  To see a generic example of how
    // to train those refer to train_shape_predictor_ex.cpp.
    deserialize("mmod_rear_end_vehicle_detector.dat") >> net >> sp;

    matrix<rgb_pixel> img;
    load_image(img, "../mmod_cars_test_image.jpg");

    image_window win;
    win.set_image(img);

    // Run the detector on the image and show us the output.
    for (auto&& d : net(img))
    {
        // We use a shape_predictor to refine the exact shape and location of the detection
        // box.  This shape_predictor is trained to simply output the 4 corner points.  So
        // all we do is make a rectangle that tightly contains those 4 points and that
        // rectangle is our refined detection position.
        auto fd = sp(img,d);
        rectangle rect;
        for (long j = 0; j < fd.num_parts(); ++j)
            rect += fd.part(j);
        win.add_overlay(rect, rgb_pixel(255,0,0));
    }



    cout << "Hit enter to view the intermediate processing steps" << endl;
    cin.get();



    // Create a tiled image pyramid and display it on the screen. 
    std::vector<rectangle> rects;
    matrix<rgb_pixel> tiled_img;
    create_tiled_pyramid<std::remove_reference<decltype(input_layer(net))>::type::pyramid_type>(img,
        tiled_img, rects, input_layer(net).get_pyramid_padding(),
        input_layer(net).get_pyramid_outer_padding());
    image_window winpyr(tiled_img, "Tiled image pyramid");



    cout << "Number of channels in final tensor image: " << net.subnet().get_output().k() << endl;
    matrix<float> network_output = image_plane(net.subnet().get_output(),0,0);
    for (long k = 1; k < net.subnet().get_output().k(); ++k)
        network_output = max_pointwise(network_output, image_plane(net.subnet().get_output(),0,k));
    const double v0_scale = img.nc()/(double)network_output.nc();
    resize_image(v0_scale, network_output);


    const float lower = -2.5;// min(network_output);
    const float upper = 0.0;// max(network_output);
    cout << "jet color mapping range:  lower="<< lower << "  upper="<< upper << endl;

    // Display the final layer as a color image
    image_window win_output(jet(network_output, upper, lower), "Output tensor from the network");



    // Overlay network_output on top of the tiled image pyramid and display it.
    matrix<rgb_pixel> tiled_img_sal = tiled_img;
    for (long r = 0; r < tiled_img_sal.nr(); ++r)
    {
        for (long c = 0; c < tiled_img_sal.nc(); ++c)
        {
            dpoint tmp(c,r);
            tmp = input_tensor_to_output_tensor(net, tmp);
            tmp = point(v0_scale*tmp);
            if (get_rect(network_output).contains(tmp))
            {
                float val = network_output(tmp.y(),tmp.x());
                rgb_alpha_pixel p;
                assign_pixel(p , colormap_jet(val,lower,upper));
                p.alpha = 120;
                assign_pixel(tiled_img_sal(r,c), p);
            }
        }
    }
    image_window win_pyr_sal(tiled_img_sal, "Saliency on image pyramid");




    // Now collapse the pyramid scales into the original image
    matrix<float> collapsed_saliency(img.nr(), img.nc());
    resizable_tensor input_tensor;
    input_layer(net).to_tensor(&img, &img+1, input_tensor);
    for (long r = 0; r < collapsed_saliency.nr(); ++r)
    {
        for (long c = 0; c < collapsed_saliency.nc(); ++c)
        {
            // Loop over a bunch of scale values and look up what part of network_output corresponds to
            // the point(c,r) in the original image, then take the max saliency value over
            // all the scales and save it at pixel point(c,r).
            float max_sal = -1e30;
            for (double scale = 1; scale > 0.2; scale *= 5.0/6.0)
            {
                // map from input image coordinates to tiled pyramid and then to output
                // tensor coordinates.
                dpoint tmp = center(input_layer(net).image_space_to_tensor_space(input_tensor,scale, drectangle(dpoint(c,r))));
                tmp = point(v0_scale*input_tensor_to_output_tensor(net, tmp));
                if (get_rect(network_output).contains(tmp))
                {
                    float val = network_output(tmp.y(),tmp.x());
                    if (val > max_sal)
                        max_sal = val;
                }
            }

            collapsed_saliency(r,c) = max_sal;

            // Also blend the saliency into the original input image so we can view it as
            // an overlay on the cars.
            rgb_alpha_pixel p;
            assign_pixel(p , colormap_jet(max_sal,lower,upper));
            p.alpha = 120;
            assign_pixel(img(r,c), p);
        }
    }

    image_window win_collapsed(jet(collapsed_saliency, upper, lower), "collapsed saliency map");
    image_window win_img_and_sal(img);


    cout << "Hit enter to end program" << endl;
    cin.get();
}
catch(image_load_error& e)
{
    cout << e.what() << endl;
    cout << "The test image is located in the examples folder.  So you should run this program from a sub folder so that the relative path is correct." << endl;
}
catch(serialization_error& e)
{
    cout << e.what() << endl;
    cout << "The model file can be obtained from: http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2   Don't forget to unzip the file." << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




