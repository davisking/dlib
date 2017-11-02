// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to run a CNN based dog face detector using dlib.  The
    example loads a pretrained model and uses it to find dog faces in images.
    We also use the dlib::shape_predictor to find the location of the eyes and
    nose and then draw glasses and a mustache onto each dog found :)
    

    Users who are just learning about dlib's deep learning API should read the
    dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp examples to learn how
    the API works.  For an introduction to the object detection method you
    should read dnn_mmod_ex.cpp


    
    TRAINING THE MODEL
        Finally, users interested in how the dog face detector was trained should
        read the dnn_mmod_ex.cpp example program.  It should be noted that the
        dog face detector used in this example uses a bigger training dataset and
        larger CNN architecture than what is shown in dnn_mmod_ex.cpp, but
        otherwise training is the same.  If you compare the net_type statements
        in this file and dnn_mmod_ex.cpp you will see that they are very similar
        except that the number of parameters has been increased.

        Additionally, the following training parameters were different during
        training: The following lines in dnn_mmod_ex.cpp were changed from
            mmod_options options(face_boxes_train, 40,40);
            trainer.set_iterations_without_progress_threshold(300);
        to the following when training the model used in this example:
            mmod_options options(face_boxes_train, 80,80);
            trainer.set_iterations_without_progress_threshold(8000);

        Also, the random_cropper was left at its default settings,  So we didn't
        call these functions:
            cropper.set_chip_dims(200, 200);
            cropper.set_min_object_size(40,40);

        The training data used to create the model is also available at 
        http://dlib.net/files/data/CU_dogs_fully_labeled.tar.gz

        Lastly, the shape_predictor was trained with default settings except we
        used the following non-default settings: cascade depth=20, tree
        depth=5, padding=0.2
*/


#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>


using namespace std;
using namespace dlib;

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
        cout << "Call this program like this:" << endl;
        cout << "./dnn_mmod_dog_hipsterizer mmod_dog_hipsterizer.dat faces/dogs.jpg" << endl;
        cout << "\nYou can get the mmod_dog_hipsterizer.dat file from:\n";
        cout << "http://dlib.net/files/mmod_dog_hipsterizer.dat.bz2" << endl;
        return 0;
    }


    // load the models as well as glasses and mustache.
    net_type net;
    shape_predictor sp;
    matrix<rgb_alpha_pixel> glasses, mustache;
    deserialize(argv[1]) >> net >> sp >> glasses >> mustache;  
    pyramid_up(glasses);
    pyramid_up(mustache);

    image_window win1(glasses);
    image_window win2(mustache);
    image_window win_wireframe, win_hipster;

    // Now process each image, find dogs, and hipsterize them by drawing glasses and a
    // mustache on each dog :)
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
        // We will also draw a wireframe on each dog's face so you can see where the
        // shape_predictor is identifying face landmarks.
        std::vector<image_window::overlay_line> lines;
        for (auto&& d : dets)
        {
            // get the landmarks for this dog's face
            auto shape = sp(img, d.rect);

            const rgb_pixel color(0,255,0);
            auto top  = shape.part(0);
            auto lear = shape.part(1);
            auto leye = shape.part(2);
            auto nose = shape.part(3);
            auto rear = shape.part(4);
            auto reye = shape.part(5);

            // The locations of the left and right ends of the mustache.
            auto lmustache = 1.3*(leye-reye)/2 + nose;
            auto rmustache = 1.3*(reye-leye)/2 + nose;

            // Draw the glasses onto the image.
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

            // Draw the mustache onto the image right under the dog's nose.
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


            // Record the lines needed for the face wire frame.
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

        cout << "Hit enter to process the next image." << endl;
        cin.get();
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




