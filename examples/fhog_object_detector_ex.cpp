// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the dlib tools for
    detecting objects in images.  In this example we will create
    three simple images, each containing some white squares.  We
    will then use the sliding window classifier tools to learn to 
    detect these squares.

*/

#include <dlib/time_this.h>

#include <dlib/image_processing/frontal_face_detector.h>

#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/array.h>
#include <dlib/array2d.h>
#include <dlib/image_keypoint.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <iostream>
#include <fstream>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main()
{  
    /*
        NOTES
        - explain the concepts of ignore boxes
    */

    try
    {
        dlib::array<array2d<unsigned char> > images, images_test;
        std::vector<std::vector<rectangle> > object_locations, object_locations_test;

        load_image_dataset(images, object_locations, "../faces/training.xml");
        upsample_image_dataset<pyramid_down<2> >(images, object_locations);

        load_image_dataset(images_test, object_locations_test, "../faces/testing.xml");
        upsample_image_dataset<pyramid_down<2> >(images_test, object_locations_test);


        add_image_left_right_flips(images, object_locations);

        cout << "num training images: " << images.size() << endl;
        cout << "num testing images:  " << images_test.size() << endl;


        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
        image_scanner_type scanner;

        scanner.set_detection_window_size(80, 80); // faces
        //scanner.set_nuclear_norm_regularization_strength(1.0);

        structural_object_detection_trainer<image_scanner_type> trainer(scanner);
        trainer.set_num_threads(6); // Set this to the number of processing cores on your machine. 
        trainer.set_c(1);
        //trainer.set_c(10);
        trainer.be_verbose();
        trainer.set_epsilon(0.01);

        // TODO, talk about this option
        //remove_unobtainable_rectangles(trainer, images, object_locations);

        object_detector<image_scanner_type> detector = trainer.train(images, object_locations);
        cout << "num filters 0.0:  "<< num_separable_filters(detector) << endl;

        cout << "training results 0.0: " << test_object_detection_function(detector, images, object_locations) << endl;
        cout << "testing results 0.0:  " << test_object_detection_function(detector, images_test, object_locations_test) << endl;

        detector = threshold_filter_singular_values(detector,0.01);
        cout << "num filters 0.01: "<< num_separable_filters(detector) << endl;
        cout << "testing results 0.01: " << test_object_detection_function(detector, images_test, object_locations_test) << endl;

        detector = threshold_filter_singular_values(detector,0.1);
        cout << "num filters 0.1: "<< num_separable_filters(detector) << endl;
        cout << "testing results 0.1: " << test_object_detection_function(detector, images_test, object_locations_test) << endl;



        image_window win, hogwin(draw_fhog(detector));
        for (unsigned long i = 0; i < images_test.size(); ++i)
        {
            std::vector<rectangle> dets;
            TIME_THIS(
            dets = detector(images_test[i]);
            );
            win.clear_overlay();
            win.set_image(images_test[i]);
            win.add_overlay(dets, rgb_pixel(255,0,0));
            cin.get();
        }


        ofstream fout("face_detector.svm", ios::binary);
        serialize(detector, fout);

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

