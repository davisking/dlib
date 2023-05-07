// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers, 
    in this case the decoding API.

    This is a pretty simple example. It loads a raw codec file, parses chunks of 
    data to the decoder and plots images to a GUI window.

    Background about video files:

        Using FFMpeg's terminology, a video/audio file has the following structure:
            - container:
                - stream 0
                - stream 1
                - stream ...
        A `container` is a file format like MP4, MP3, WAV.
        A `stream` is encoded media like video, audio ( or subtitles) 
        using a codec like H264, H265, VP9, AAC, A3C, etc.

        MP4 isn't a codec and H264 isn't (strictly speaking) a file format. 
        The first describes a packet structure for saving encoded streams to file, 
        it contains header information, trailer information, and describes how to 
        interleave multiple streams in a file.
        The later is a protocol for compressing raw media streams into something smaller in size, 
        suitable for saving to file, transmitting over a network connection or adding
        to a `container` file.
        Note, FFMpeg treats network protocols like HTTP, RTMP, RTSP as containers.

        Dlib's dlib::ffmpeg::demuxer class reads `container` files like MP4, MP3 or RTSP streams, 
        extracts and decodes each stream.

        Dlib's dlib::ffmpeg::decoder class reads raw encoded DATA like H264 or PCM data
        and decodes it to images or audio frames.
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
    parser.add_option("i",      "input video encoded stream. e.g. dlib/test/ffmpeg_data/MOT20-08-raw.h264", 1);
    parser.add_option("codec",  "codec name. e.g. h264", 1);
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

    const std::string filepath = parser.option("i").argument();
    const std::string codec    = parser.option("codec").argument();

    decoder dec([&] {
        decoder::args args;
        args.codec_name = codec;
        return args;
    }());

    if (!dec.is_open())
    {
        printf("Failed to create decoder.\n");
        return EXIT_FAILURE;
    }

    image_window win;

    const auto callback = [&](array2d<rgb_pixel>& img)
    {
        win.set_image(img);
    };

    ifstream fin{filepath, std::ios::binary};
    std::vector<char> buf(1024);

    while (fin)
    {
        fin.read(buf.data(), buf.size());
        size_t ret = fin.gcount();
        dec.push((const uint8_t*)buf.data(), ret, wrap(callback));
    }

    dec.flush(wrap(callback));

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}