// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the Bulk Synchronous Parallel (BSP)
    processing tools from the dlib C++ Library.  These tools allow you to easily setup a
    number of processes running on different computers which cooperate to compute some
    result.  

*/

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>


using namespace dlib;
using namespace std;

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "Call this program like this: " << endl;
        cout << "./video_tracking_ex ../video_frames" << endl;
        return 1;
    }

    std::vector<file> files = get_files_in_directory_tree(argv[1], match_ending(".jpg"));
    std::sort(files.begin(), files.end());
    if (files.size() == 0)
    {
        cout << "No images found in " << argv[1] << endl;
        return 1;
    }

    array2d<unsigned char> img;
    load_image(img, files[0]);

    correlation_tracker tracker;
    tracker.start_track(img, centered_rect(point(93,110), 38, 86));

    image_window win;
    for (unsigned long i = 1; i < files.size(); ++i)
    {
        load_image(img, files[i]);
        tracker.update(img);

        win.set_image(img); 
        win.clear_overlay(); 
        win.add_overlay(tracker.get_position());

        cout << "hit enter to process next frame" << endl;
        cin.get();
    }
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

