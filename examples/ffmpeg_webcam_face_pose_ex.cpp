// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use dlib's demuxer object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/gui_widgets.h>
#include <dlib/media.h>

using namespace std;
using namespace dlib;
using namespace dlib::ffmpeg;

int main(int argc, const char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("height",     "height of frames", 1);
        parser.add_option("width",      "width of frames", 1);
        parser.add_option("framerate",  "webcam desired framerate", 1);
        parser.set_group_name("Help Options");
        parser.add_option("h",          "alias of --help");
        parser.add_option("help",       "display this message and exit");

        parser.parse(argc, argv);
        const char* one_time_opts[] = {"height", "width", "framerate"};
        parser.check_one_time_options(one_time_opts);

        if (parser.option("h") || parser.option("help"))
        {
            parser.print_options();
            cout << "Please use `v4l2-ctl --list-formats-ext` to view all supported hardware formats\n";
            return 0;
        }

        demuxer cap{[&]
        {
            ffmpeg::demuxer::args args;
            args.filepath   = "/dev/video0";
            args.height     = get_option(parser, "height", 0);
            args.width      = get_option(parser, "width",  0);
            args.framerate  = get_option(parser, "framerate", 0);
            return args;
        }()};

        if (!cap.is_open())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        cout << "height  : " << cap.height() << '\n';
        cout << "width   : " << cap.width() << '\n';
        cout << "fps     : " << cap.fps() << '\n';
        
        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        array2d<rgb_pixel> img;

        // Grab and process frames until the main window is closed by the user.
        while(cap.read(img) && !win.is_closed())
        {
            // Detect faces 
            std::vector<rectangle> faces = detector(img);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(img, faces[i]));

            // Display it all on the screen
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(render_face_detections(shapes));
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

