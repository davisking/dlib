

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



    // shape predictor was trained with these settings: tree cascade depth=20, tree depth=5, padding=0.2
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
    shape_predictor sp;
    matrix<rgb_alpha_pixel> glasses, mustache;
    deserialize(argv[1]) >> net >> sp >> glasses >> mustache;  
    pyramid_up(glasses);
    pyramid_up(mustache);

    // right eye (59,35),  left eye (176,36)
    image_window win1(glasses);
    image_window win2(mustache);

    image_window win_wireframe, win_hipster;
    for (int i = 2; i < argc; ++i)
    {
        matrix<rgb_pixel> img;
        load_image(img, argv[i]);

        // Upsampling the image will allow us to find smaller dog faces but will use more
        // computational resources.
        //pyramid_up(img); 

        auto dets = net(img);
        win_wireframe.clear_overlay();
        win_wireframe.set_image(img);
        std::vector<image_window::overlay_line> lines;
        for (auto&& d : dets)
        {
            auto shape = sp(img, d.rect);

            const rgb_pixel color(0,255,0);
            auto top  = shape.part(0);
            auto lear = shape.part(1);
            auto leye = shape.part(2);
            auto nose = shape.part(3);
            auto rear = shape.part(4);
            auto reye = shape.part(5);

            auto lmustache = 1.3*(leye-reye)/2 + nose;
            auto rmustache = 1.3*(reye-leye)/2 + nose;

            std::vector<point> from = {2*point(176,36), 2*point(59,35)}, to = {leye, reye};
            auto tform = find_similarity_transform(from, to);
            for (long r = 0; r < glasses.nr(); ++r)
            {
                for (long c = 0; c < glasses.nc(); ++c)
                {
                    point p = tform(point(c,r));
                    if (get_rect(img).contains(p))
                        assign_pixel(img(p.y(),p.x()), glasses(r,c));
                }
            }
            auto mrect = get_rect(mustache);
            from = {mrect.tl_corner(), mrect.tr_corner()};
            to = {rmustache, lmustache};
            tform = find_similarity_transform(from, to);
            for (long r = 0; r < mustache.nr(); ++r)
            {
                for (long c = 0; c < mustache.nc(); ++c)
                {
                    point p = tform(point(c,r));
                    if (get_rect(img).contains(p))
                        assign_pixel(img(p.y(),p.x()), mustache(r,c));
                }
            }


            lines.push_back(image_window::overlay_line(leye, nose, color));
            lines.push_back(image_window::overlay_line(nose, reye, color));
            lines.push_back(image_window::overlay_line(reye, leye, color));
            lines.push_back(image_window::overlay_line(reye, rear, color));
            lines.push_back(image_window::overlay_line(rear, top, color));
            lines.push_back(image_window::overlay_line(top, lear,  color));
            lines.push_back(image_window::overlay_line(lear, leye,  color));
        }

        win_wireframe.add_overlay(lines);
        win_hipster.set_image(img);

        cin.get();
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




