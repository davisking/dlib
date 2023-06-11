// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example that illustrates how to use the ffmpeg wrappers
    for taking screen grabs and plotting to a GUI window.
*/

#include <cstdio>
#include <dlib/media.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;

int main()
try
{
    const auto demuxers              = ffmpeg::list_demuxers();
    const bool screen_grab_available = std::find(begin(demuxers), end(demuxers), "x11grab") != demuxers.end();

    if (screen_grab_available)
    {
        ffmpeg::demuxer::args args;
        args.filepath       = "";
        args.input_format   = "x11grab";

        ffmpeg::demuxer cap(args);
        if (!cap.is_open() || !cap.video_enabled())
        {
            printf("Failed to open demuxer for screen grab\n");
            return EXIT_FAILURE;
        }

        image_window win;

        ffmpeg::frame frame;
        array2d<rgb_pixel> img;

        while (cap.read(frame))
        {
            convert(frame, img);
            win.set_image(img);
        }
    }
    else
    {
        printf("Sorry your installation of ffmpeg doesn't support screen grab\n");
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}