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

    image_window win;

    /*
        For simplicity we use the constructor which takes a filepath only. All other parameters are defaulted or guessed.
        Equivalently, we could have done:

            ffmpeg::demuxer::args args;
            args.filepath = filepath;
            ffmpeg::demuxer cap(args);

        Furthermore, we can set additional settings in args, for example:

            // This disables extracting and decoding images. 
            // You may want to do this if you're only interesting in extracting audio
            args.enable_image = false; 

            // This disables extracting and decoding audio.
            // You may want to do this if you don't care about audio in your video. 
            // This saves processing time and you don't have to deal with audio frame objects in your code.
            args.enable_audio = false;

            // This will resize frames before presenting them to the user. 
            // I.e. frames "returned" by demuxer::read() will have this height.
            // By default, the demuxer object does not resize frames.
            args.image_options.h = SOME_HEIGHT;

            // Same as above but for width.
            args.image_options.w = SOME_WIDTH;

            // By default, demuxer reformats frames from the default format in the encoded stream to RGB.
            // You can set this to AV_PIX_FMT_NONE and demuxer will leave frames in their default format.
            // This is likely to be AV_PIX_FMT_YUV420P.
            // However, you can set it to anything that FFMPEG supports, and frames will be presented
            // in that format.
            args.image_options.fmt = SOME_OTHER_PIXEL_FORMAT;

            // Same as above, by default, demuxer leaves audio frames in their default sample rate.
            // But user can change this, and audio will be resampled to that rate.
            // Note, reducing the sample rate reduces the quality of the audio.
            // You can artificially upsample audio, but it won't make the quality any better.
            args.audio_options.sample_rate = SOME_SAMPLE_RATE;

            // You may want to do this if you want more or less channels.
            // Note, dlib only has one audio object "audio_frame", which is stereo and uses int16_t sample format.
            // So if you're going to use other layouts and sample formats, you won't be able to use audio_frame.
            // You will have to use ffmpeg::frame directly. Use with care and please visit ffmpeg's documentation.
            args.audio_options.channel_layout = SOME_OTHER_LAYOUT; // e.g. AV_CH_LAYOUT_MONO, AV_CH_LAYOUT_STEREO. See libavutil/channel_layout.h

            // This changes the default sample format.
            args.audio_options.fmt = SOME_OTHER_SAMPLE_FORMAT;
    */

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

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}