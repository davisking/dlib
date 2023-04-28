// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers, 
    in this case the demuxer API.

    In this example, we show how to read both images and audio.
*/

#include <cstdio>
#include <dlib/media.h>
#include <dlib/gui_widgets.h>
#include <dlib/cmd_line_parser.h>

using namespace std;
using namespace dlib;
using namespace dlib::ffmpeg;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("i",       "input video", 1);
    parser.add_option("verbose", "enable all internal ffmpeg logging");
    parser.set_group_name("Help Options");
    parser.add_option("h",       "alias of --help");
    parser.add_option("help",    "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    const std::string filepath = parser.option("i").argument();

    // We use the most simple constructor, the one that takes in filepath.
    // By default, both images and audio are read.
    demuxer cap(filepath);
    if (!cap.is_open())
    {
        printf("%s is not a valid video file\n", filepath.c_str());
        return EXIT_FAILURE;
    }

    // Read images and audio.
    // Images are plotted, audio samples are counted.
    frame               f;
    array2d<rgb_pixel>  img;
    image_window        win;
    int                 nsamples{0};

    // When reading frames, we get exactly what's in the codec by default.
    // To resize, change pixel format, resample or change sample format, 
    // you have to pass extra arguments to read() which either resize or resample
    // the frame. Since we want rgb_pixel, we need to set the pixel format appropriately.
    const resizing_args args_image {0, 0, pix_traits<rgb_pixel>::fmt};

    while (cap.read(f, args_image))
    {
        if (f.is_image())
        {
            convert(f, img);
            win.set_image(img);
        }
        else if (f.is_audio())
        {
            nsamples += f.nsamples();
        }
    }
    
    printf("Read %i audio samples\n", nsamples);

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}