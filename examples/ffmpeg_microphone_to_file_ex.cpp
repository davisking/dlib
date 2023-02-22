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
    parser.add_option("t", "capture time in seconds", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"t"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    const seconds time{get_option(parser, "t",  1)};

    // We're going to look for an appropriate audio device.
    // On linux, look for a device type "alsa"

    const string device = []
    {
        const auto devices = ffmpeg::list_input_devices();

        for (auto&& info : devices)
            if (info.device_type == "alsa")
                for (auto&& instance : info.devices)
                    if (instance.name.find("hw:") != string::npos)
                        return "hw:0,0";

        return "";
    }();

    if (device.empty())
    {
        cout << "Didn't find a microphone. Exiting.\n";
        return EXIT_FAILURE;
    }

    // Open microphone
    demuxer cap([&] {
        demuxer::args args;
        args.filepath       = device;
        args.input_format   = "alsa";
        return args;
    }());

    if (!cap.is_open())
    {
        cout << "Failed to open " << device << endl;
        return EXIT_FAILURE;
    }

    // Create WAV file
    muxer writer([&] {
        muxer::args args;
        args.filepath                   = "recording.wav";
        args.enable_image               = false;
        args.args_audio.codec_name      = "pcm_s16le";
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

    // Pull and push]
    const auto start = high_resolution_clock::now();
    frame f;
    while (cap.read(f) && (high_resolution_clock::now() - start) < time)
        writer.push(std::move(f));

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}