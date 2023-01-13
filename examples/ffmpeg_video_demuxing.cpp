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

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("i", "input video", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias of --help");
    parser.add_option("help", "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    const std::string filepath = get_option(parser, "i", "");

    image_window win;

    // For simplicity we use the constructor which takes a filepath only. All other parameters are defaulted or guessed.
    // We could have instead done this:
    //      ffmpeg::demuxer::args args;
    //      args.filepath = filepath;
    //      ffmpeg::demuxer cap(args);

    ffmpeg::demuxer cap(filepath);
    if (!cap.is_open())
    {
        printf("%s is not a valid video file\n", filepath.c_str());
        return EXIT_FAILURE;
    }

    printf("Video properties:\n\n");
    printf("Estimated duration      : %f\n", cap.duration());
    printf("Video contains images   : %i\n", cap.video_enabled());
    if (cap.video_enabled())
    {
        printf("    height              : %i\n", cap.height());
        printf("    width               : %i\n", cap.width());
        printf("    pixel format        : %s\n", ffmpeg::get_pixel_fmt_str(cap.pixel_fmt()).c_str());
        printf("    fps                 : %f\n", cap.fps());
        printf("    nframes             : %d\n", cap.estimated_nframes());
        printf("    codec               : %s\n", cap.get_video_codec_name().c_str());
    }
    printf("Video contains audio : %i\n", cap.audio_enabled());
    if (cap.audio_enabled())
    {
        printf("    sample rate         : %i\n", cap.sample_rate());
        printf("    channel layout      : %s\n", ffmpeg::get_channel_layout_str(cap.channel_layout()).c_str());
        printf("    sample format       : %s\n", ffmpeg::get_audio_fmt_str(cap.sample_fmt()).c_str());
        printf("    nchannels           : %i\n", cap.nchannels());
        printf("    estimated samples   : %i\n", cap.estimated_total_samples());
        printf("    codec               : %s\n", cap.get_audio_codec_name().c_str());
    }

    printf("\n\n");
    printf("Video metadata:\n\n");
   
    for (auto&& stream_metadata_pair : cap.get_all_metadata())
    {
        printf("    Stream %i\n", stream_metadata_pair.first);
        for (auto&& metadata : stream_metadata_pair.second)
            printf("        key `%s` : val `%s`\n", metadata.first.c_str(), metadata.second.c_str());
    }

    ffmpeg::frame frame;
    array2d<rgb_pixel> img;

    while (cap.read(frame))
    {
        convert(frame, img);
        win.set_image(img);
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}