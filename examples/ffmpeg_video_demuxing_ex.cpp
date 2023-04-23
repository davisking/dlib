// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers, 
    in this case the demuxer API.

    This is a pretty simple example. It loads a video file, and plots
    the image frames on a GUI window.
*/

#include <cstdio>
#include <dlib/media.h>
#include <dlib/gui_widgets.h>
#include <dlib/cmd_line_parser.h>

using namespace std;
using namespace dlib;
using namespace dlib::ffmpeg;

void print_properties(const demuxer& cap)
{
    printf("Video properties:\n\n");
    printf("Estimated duration      : %f\n", cap.duration());
    printf("Video contains images   : %i\n", cap.video_enabled());
    if (cap.video_enabled())
    {
        printf("    height              : %i\n", cap.height());
        printf("    width               : %i\n", cap.width());
        printf("    pixel format        : %s\n", get_pixel_fmt_str(cap.pixel_fmt()).c_str());
        printf("    fps                 : %f\n", cap.fps());
        printf("    nframes             : %d\n", cap.estimated_nframes());
        printf("    codec               : %s\n", cap.get_video_codec_name().c_str());
    }
    printf("Video contains audio    : %i\n", cap.audio_enabled());
    if (cap.audio_enabled())
    {
        printf("    sample rate         : %i\n", cap.sample_rate());
        printf("    channel layout      : %s\n", get_channel_layout_str(cap.channel_layout()).c_str());
        printf("    sample format       : %s\n", get_audio_fmt_str(cap.sample_fmt()).c_str());
        printf("    nchannels           : %i\n", cap.nchannels());
        printf("    estimated samples   : %i\n", cap.estimated_total_samples());
        printf("    codec               : %s\n", cap.get_audio_codec_name().c_str());
    }

    printf("\n\n");
    printf("Video metadata:\n");
   
    for (auto&& metadata : cap.get_metadata())
        printf("    key : %-32s ; val : %-32s\n", metadata.first.c_str(), metadata.second.c_str());
}

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

    if (parser.option("verbose"))
    {
        // You can set the verbosity of some global loggers:
        //  - logger_dlib_wrapper() is the global logger used by dlib's wrappers
        //  - logger_ffmpeg() is the global logger used by the internal ffmpeg libraries (Nice that we are able to do that!)
        ffmpeg::logger_dlib_wrapper().set_level(LALL);
        ffmpeg::logger_ffmpeg().set_level(LALL);
    }

    const std::string filepath = parser.option("i").argument();

    // In this example, we only read images.
    // This constructor allows you to concisely enabled images and disable audio frames.
    demuxer cap({filepath, video_enabled, audio_disabled});
    if (!cap.is_open())
    {
        printf("%s is not a valid video file\n", filepath.c_str());
        return EXIT_FAILURE;
    }

    // Print all the demuxer's properties for fun
    print_properties(cap);

    // Read all images and plot
    array2d<rgb_pixel> img;
    image_window win;

    while (cap.read(img))
        win.set_image(img);

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}