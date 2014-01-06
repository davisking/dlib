// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image.  In
    particular, this program shows how you can take a list of images from the
    command line and display each on the screen with red boxes overlaid on each
    human face.

    The examples/faces folder contains some jpg images of people.  You can run
    this program on them and see the detections by executing the following:
        ./face_detection_ex faces/*.jpg

    
    This face detector is made using the now classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  This type of object detector is fairly
    general and capable of detecting many types of semi-rigid objects in
    addition to human faces.  Therefore, if you are interested in making your
    own object detectors then read the fhog_object_detector_ex.cpp example
    program.  It shows how to use the machine learning tools used to create this
    face detector. 


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2.  If you are using cmake to
    compile this program you can enable them by using one of the following
    commands when you create the build project:
        cmake path_to_dclib/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dclib/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dclib/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  

*/


#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <iostream>
#include <fstream>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  

    try
    {
        if (argc != 2)
        {
            cout << "Give the path to the dclib/examples/faces directory as the argument to this" << endl;
            cout << "program.  For example, if you are in the dclib/examples folder then execute " << endl;
            cout << "this program by running: " << endl;
            cout << "   ./fhog_object_detector_ex faces" << endl;
            cout << endl;
            return 0;
        }
        const std::string faces_directory = argv[1];

        dlib::array<array2d<unsigned char> > images, images_test;
        std::vector<std::vector<rectangle> > object_locations, object_locations_test;

        /*
            These xml files are created by the imglab tool.
            To create this annotated data you will need to use the imglab tool 
            included with dlib.  It is located in the tools/imglab folder and can be compiled
            using the following commands.  
                cd tools/imglab
                mkdir build
                cd build
                cmake ..
                cmake --build . --config Release
            Note that you may need to install CMake (www.cmake.org) for this to work.  

            Next, lets assume you have a folder of images called /tmp/images.  These images 
            should contain examples of the objects you want to learn to detect.  You will 
            use the imglab tool to label these objects.  Do this by typing the following
                ./imglab -c mydataset.xml /tmp/images
            This will create a file called mydataset.xml which simply lists the images in 
            /tmp/images.  To annotate them run
                ./imglab mydataset.xml
            A window will appear showing all the images.  You can use the up and down arrow 
            keys to cycle though the images and the mouse to label objects.  In particular, 
            holding the shift key, left clicking, and dragging the mouse will allow you to 
            draw boxes around the objects you wish to detect.  So next, label all the objects 
            with boxes.  Note that it is important to label all the objects since any object 
            not labeled is implicitly assumed to be not an object we should detect.

            Once you finish labeling objects go to the file menu, click save, and then close 
            the program. This will save the object boxes back to mydataset.xml.  You can verify 
            this by opening the tool again with
                ./imglab mydataset.xml
            and observing that the boxes are present.
        */
        load_image_dataset(images, object_locations, faces_directory+"/training.xml");
        load_image_dataset(images_test, object_locations_test, faces_directory+"/testing.xml");
        upsample_image_dataset<pyramid_down<2> >(images,      object_locations);
        upsample_image_dataset<pyramid_down<2> >(images_test, object_locations_test);
        add_image_left_right_flips(images, object_locations);
        cout << "num training images: " << images.size() << endl;
        cout << "num testing images:  " << images_test.size() << endl;

        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
        image_scanner_type scanner;
        scanner.set_detection_window_size(80, 80); 
        structural_object_detection_trainer<image_scanner_type> trainer(scanner);
        trainer.set_num_threads(4); // Set this to the number of processing cores on your machine. 
        trainer.set_c(1);
        trainer.be_verbose();
        // stop when the risk gap is less than 0.01
        trainer.set_epsilon(0.01);

        // TODO, talk about this option. 
        //remove_unobtainable_rectangles(trainer, images, object_locations);

        object_detector<image_scanner_type> detector = trainer.train(images, object_locations);

        // prints the precision, recall, and average precision 
        cout << "training results: " << test_object_detection_function(detector, images, object_locations) << endl;
        cout << "testing results:  " << test_object_detection_function(detector, images_test, object_locations_test) << endl;


        image_window hogwin(draw_fhog(detector), "Learned fHOG filter");

        image_window win; 
        for (unsigned long i = 0; i < images_test.size(); ++i)
        {
            std::vector<rectangle> dets = detector(images_test[i]);
            // Now we show the image on the screen and the face detections as
            // red overlay boxes.
            win.clear_overlay();
            win.set_image(images_test[i]);
            win.add_overlay(dets, rgb_pixel(255,0,0));

            cout << "Hit enter to process the next image..." << endl;
            cin.get();
        }


        ofstream fout("face_detector.svm", ios::binary);
        serialize(detector, fout);
        fout.close();

        ifstream fin("face_detector.svm", ios::binary);
        object_detector<image_scanner_type> detector2;
        deserialize(detector2, fin);



        /*
            Advanced features...
            - explain the concepts of ignore boxes
            - talk about putting multiple detectors inside a single object_detector object.  
        */

        // talk about low nuclear norm stuff
        //scanner.set_nuclear_norm_regularization_strength(1.0);
        detector = threshold_filter_singular_values(detector,0.1);
        cout << "num filters 0.0:  "<< num_separable_filters(detector) << endl;

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

