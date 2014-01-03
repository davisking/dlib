// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*


*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        frontal_face_detector detector = get_frontal_face_detector();
        image_window win;
        for (int i = 1; i < argc; ++i)
        {
            array2d<unsigned char> img;
            load_image(img, argv[i]);
            pyramid_up(img);
            std::vector<rectangle> dets = detector(img);

            cout << "number of faces detected: " << dets.size() << endl;
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(dets, rgb_pixel(255,0,0));
            // Pause until the user hits the enter key
            cin.get();
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

