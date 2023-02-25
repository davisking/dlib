// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers.
    It attempts to read audio from a microphone if available, and saves the audio to wav.
*/

#include <iostream>
#include <chrono>
#include <dlib/media.h>
#include <dlib/cmd_line_parser.h>

using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;
using namespace dlib;
using namespace dlib::ffmpeg;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("i",      "input audio file", 1);
    parser.add_option("codec",  "audio codec. E.g. pcm_s16le", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i", "codec"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    if (!parser.option("i"))
    {
        cout << "Missing -i" << endl;
        parser.print_options();
        return 0;
    }
    
    const std::string filename  = get_option(parser, "i",       "");
    const std::string codec     = get_option(parser, "codec",   "pcm_s16le");

    // We're going to look for an appropriate audio device.
    // On linux, look for a device type "alsa"

    const string device = []
    {
        const auto devices = ffmpeg::list_output_devices();

        for (auto&& info : devices)
            if (info.device_type == "alsa")
                for (auto&& instance : info.devices)
                    if (instance.name.find("hw:") != string::npos)
                        return "hw:0,0";

        return "";
    }();

    if (device.empty())
    {
        cout << "Didn't find a speaker. Exiting.\n";
        return EXIT_FAILURE;
    }

    // Open file
    demuxer cap({filename, video_disabled, audio_enabled});

    if (!cap.is_open())
    {
        cout << "Failed to open " << device << endl;
        return EXIT_FAILURE;
    }

    // Create writer to speaker
    muxer writer([&] {
        muxer::args args;
        args.filepath                   = device;
        args.output_format              = "alsa";
        args.enable_image               = false;
        args.args_audio.codec_name      = codec;
        args.args_audio.sample_rate     = cap.sample_rate();
        args.args_audio.channel_layout  = cap.channel_layout();
        args.args_audio.fmt             = cap.sample_fmt();
        return args;
    }());

    if (!writer.is_open())
    {
        cout << "Failed to open wav file" << endl;
        return EXIT_FAILURE;
    }

    // Pull and push
    frame f;
    while (cap.read(f))
        writer.push(std::move(f));

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}