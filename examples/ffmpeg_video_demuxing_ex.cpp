// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers, 
    in this case the demuxer API.

    This example loads a video in two ways:
        - A simplified method which only allows reading images from a video, and reads directly to dlib image objects.
        - A more advanced example which can read both images and audio and uses the ffmpeg::frame object, which is
          a type-erased frame buffer. This is then converted to a dlib image object if it contains an image.
*/

#include <cstdio>
#include <dlib/media.h>
#include <dlib/gui_widgets.h>
#include <dlib/cmd_line_parser.h>

using namespace std;
using namespace dlib;
using namespace dlib::ffmpeg;

void method_basic(const std::string& filepath)
{
    demuxer cap({filepath, video_enabled, audio_disabled});

    image_window win;
    array2d<rgb_pixel> img;

    while (cap.read(img))
        win.set_image(img);
}

void method_advanced(const std::string& filepath)
{
    ffmpeg::demuxer cap(filepath);

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
    printf("Video contains audio    : %i\n", cap.audio_enabled());
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
    printf("Video metadata:\n");
   
    for (auto&& metadata : cap.get_metadata())
        printf("    key : %-32s ; val : %-32s\n", metadata.first.c_str(), metadata.second.c_str());

    image_window win;
    ffmpeg::frame frame;
    array2d<rgb_pixel> img;
    size_t audio_samples{0};

    while (cap.read(frame))
    {
        if (frame.is_image() && frame.pixfmt() == AV_PIX_FMT_RGB24)
        {
            convert(frame, img);
            win.set_image(img);
        }

        if (frame.is_audio())
        {
            audio_samples += frame.nsamples();
            printf("\r\tDecoding %zu samples", audio_samples); fflush(stdout);
        }
    }

    printf("\n");
}

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("i",      "input video", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    const std::string filepath = get_option(parser, "i", "");

    method_basic(filepath);
    method_advanced(filepath);

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}