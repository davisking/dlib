// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the dlib::rdrand kernel
    in the dlib C++ Library.  It is a simple tool for displaying some 
    digital random noise on the screen.

*/

#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
 
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main()
{
    uint16 size = 1024;
    matrix<unsigned char> img(size, size);

    dlib::rdrand rdrnd;
    
    for (uint16 r = 0; r < img.nr(); r++)
    {
        for (uint16 c = 0; c < img.nc(); c++)
        {
            unsigned char color = rdrnd.get_random_8bit_number();
            assign_pixel(img(r, c), color);
        }
    }

    image_window win(img, "digital random noise");

    win.wait_until_closed();
}

//  ----------------------------------------------------------------------------

