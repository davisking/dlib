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
    parser.add_option("t",      "capture time in seconds", 1);
    parser.add_option("i",      "input microphone device. E.g. hw:0,0", 1);
    parser.add_option("o",      "output audio file. E.g. recording.m4a, recording.wav. Default: recording.m4a", 1);
    parser.add_option("codec",  "audio codec. E.g. `aac`, `pcm_s16le`. Recommend `pcm_s16le` for WAV files and `aac` for `M4A` files. Default: `aac`", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"t", "i", "o", "codec"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    const seconds time{get_option(parser, "t",  1)};
    const std::string device    = get_option(parser, "i",       "hw:0,0");
    const std::string filename  = get_option(parser, "o",       "recording.m4a");
    const std::string codec     = get_option(parser, "codec",   "aac");

    // Open microphone
    demuxer cap([&] {
        demuxer::args args;
        args.filepath       = device;
        args.input_format   = "alsa";
        return args;
    }());

    if (!cap.is_open())
    {
        cout << "Failed to open device: " << device << '\n';
        return EXIT_FAILURE;
    }

    // Create WAV file
    muxer writer([&] {
        muxer::args args;
        args.filepath                   = filename;
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